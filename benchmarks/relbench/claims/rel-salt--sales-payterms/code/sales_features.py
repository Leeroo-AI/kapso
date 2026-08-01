import time

import duckdb
import numpy as np
import pandas as pd


HEADER_COLUMNS = [
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
]

ROLE_COLUMNS = ["PAYERPARTY", "BILLTOPARTY", "SOLDTOPARTY", "SHIPTOPARTY"]

CATEGORICAL_COLUMNS = HEADER_COLUMNS + [
    "CATEGORY_TOP1",
    "CATEGORY_TOP2",
    "CATEGORY_TOP3",
    "PRODUCT_TOP1",
    "PRODUCT_TOP2",
    "PRODUCT_TOP3",
] + [f"{role}_{geo}" for role in ROLE_COLUMNS for geo in ("COUNTRY", "REGION")]


def extract_documents(ctx):
    started = time.time()
    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 11")
    con.execute("SET memory_limit = '16GB'")
    for name in ("salesdocument", "salesdocumentitem", "customer", "address"):
        con.register(name, ctx.db.table_dict[name].df)
    temporal = con.execute(
        """
        SELECT count(*)
        FROM salesdocumentitem i
        JOIN salesdocument d USING (SALESDOCUMENT)
        WHERE i.CREATIONTIMESTAMP > d.CREATIONTIMESTAMP
        """
    ).fetchone()[0]
    if temporal:
        raise RuntimeError(f"temporal item assertion failed for {temporal} rows")
    item_agg = con.execute(
        """
        SELECT
            SALESDOCUMENT,
            count(*)::INTEGER AS ITEM_COUNT,
            count(DISTINCT PRODUCT)::INTEGER AS PRODUCT_DISTINCT,
            count(DISTINCT SALESDOCUMENTITEMCATEGORY)::INTEGER AS CATEGORY_DISTINCT,
            mode(PAYERPARTY ORDER BY ID) AS PAYERPARTY,
            mode(BILLTOPARTY ORDER BY ID) AS BILLTOPARTY,
            mode(SOLDTOPARTY ORDER BY ID) AS SOLDTOPARTY,
            mode(SHIPTOPARTY ORDER BY ID) AS SHIPTOPARTY,
            count(DISTINCT PAYERPARTY)::INTEGER AS PAYERPARTY_DISTINCT,
            count(DISTINCT BILLTOPARTY)::INTEGER AS BILLTOPARTY_DISTINCT,
            count(DISTINCT SOLDTOPARTY)::INTEGER AS SOLDTOPARTY_DISTINCT,
            count(DISTINCT SHIPTOPARTY)::INTEGER AS SHIPTOPARTY_DISTINCT
        FROM salesdocumentitem
        GROUP BY SALESDOCUMENT
        """
    ).fetchdf()
    print(f"[phase] extraction item_agg={len(item_agg)}")

    def composition_summary(source, prefix):
        return con.execute(
            f"""
            WITH counts AS (
                SELECT SALESDOCUMENT, {source} AS val, count(*)::INTEGER AS n
                FROM salesdocumentitem
                GROUP BY SALESDOCUMENT, {source}
            )
            SELECT
                SALESDOCUMENT,
                (list(val ORDER BY n DESC, val ASC))[1] AS {prefix}_TOP1,
                (list(val ORDER BY n DESC, val ASC))[2] AS {prefix}_TOP2,
                (list(val ORDER BY n DESC, val ASC))[3] AS {prefix}_TOP3,
                max(n)::INTEGER AS {prefix}_TOP_COUNT,
                sum(n * ln(n)) AS {prefix}_NLOGN,
                bit_xor(DISTINCT hash(val)) AS {prefix}_SET_HASH
            FROM counts
            GROUP BY SALESDOCUMENT
            """
        ).fetchdf()

    category_summary = composition_summary("SALESDOCUMENTITEMCATEGORY", "CATEGORY")
    print(f"[phase] extraction category_summary={len(category_summary)}")
    product_summary = composition_summary("PRODUCT", "PRODUCT")
    print(f"[phase] extraction product_summary={len(product_summary)}")
    customer_geo = con.execute(
        """
        SELECT c.CUSTOMER, a.COUNTRY, a.REGION
        FROM customer c
        LEFT JOIN address a USING (ADDRESSID)
        """
    ).fetchdf()
    print(f"[phase] extraction customer_geo={len(customer_geo)}")
    documents = ctx.db.table_dict["salesdocument"].df[
        ["SALESDOCUMENT", "CREATIONTIMESTAMP", *HEADER_COLUMNS]
    ].copy()
    documents = documents.merge(item_agg, on="SALESDOCUMENT", how="inner", validate="one_to_one")
    documents = documents.merge(category_summary, on="SALESDOCUMENT", how="left", validate="one_to_one")
    documents = documents.merge(product_summary, on="SALESDOCUMENT", how="left", validate="one_to_one")
    print(f"[phase] extraction composition_join={len(documents)}")
    for role in ROLE_COLUMNS:
        geo = customer_geo.rename(
            columns={
                "CUSTOMER": role,
                "COUNTRY": f"{role}_COUNTRY",
                "REGION": f"{role}_REGION",
            }
        )
        documents = documents.merge(geo, on=role, how="left", validate="many_to_one")
    print(f"[phase] extraction geography_join={len(documents)}")
    con.close()
    documents["CATEGORY_ENTROPY"] = np.log(documents["ITEM_COUNT"]) - (
        documents["CATEGORY_NLOGN"] / documents["ITEM_COUNT"]
    )
    documents["PRODUCT_ENTROPY"] = np.log(documents["ITEM_COUNT"]) - (
        documents["PRODUCT_NLOGN"] / documents["ITEM_COUNT"]
    )
    documents = documents.drop(columns=["CATEGORY_NLOGN", "PRODUCT_NLOGN"])
    documents["CREATIONTIMESTAMP"] = pd.to_datetime(documents["CREATIONTIMESTAMP"])
    for column in CATEGORICAL_COLUMNS:
        values = documents[column].astype("string").fillna("__MISSING__")
        documents[column] = pd.factorize(values, sort=True)[0].astype(np.int32) + 1
    for column in ROLE_COLUMNS:
        documents[column] = documents[column].fillna(-1).astype(np.int64)
        distinct = f"{column}_DISTINCT"
        documents[f"{column}_AMBIGUOUS"] = (documents[distinct] > 1).astype(np.int8)
    for left_index, left in enumerate(ROLE_COLUMNS):
        for right in ROLE_COLUMNS[left_index + 1 :]:
            documents[f"{left}_EQ_{right}"] = (documents[left] == documents[right]).astype(np.int8)
            for geo in ("COUNTRY", "REGION"):
                documents[f"{left}_{geo}_EQ_{right}"] = (
                    documents[f"{left}_{geo}"] == documents[f"{right}_{geo}"]
                ).astype(np.int8)
    country_columns = [f"{role}_COUNTRY" for role in ROLE_COLUMNS]
    region_columns = [f"{role}_REGION" for role in ROLE_COLUMNS]
    documents["ROLE_COUNTRY_DISTINCT"] = documents[country_columns].nunique(axis=1).astype(np.int8)
    documents["ROLE_REGION_DISTINCT"] = documents[region_columns].nunique(axis=1).astype(np.int8)
    timestamp = documents["CREATIONTIMESTAMP"]
    documents["CALENDAR_YEAR"] = timestamp.dt.year.astype(np.int16)
    documents["CALENDAR_MONTH"] = timestamp.dt.month.astype(np.int8)
    documents["CALENDAR_DAY"] = timestamp.dt.day.astype(np.int8)
    documents["CALENDAR_DOW"] = timestamp.dt.dayofweek.astype(np.int8)
    documents["CALENDAR_HOUR"] = timestamp.dt.hour.astype(np.int8)
    documents["DAYS_FROM_2018"] = (
        (timestamp - pd.Timestamp("2018-01-01")) / pd.Timedelta(days=1)
    ).astype(np.float32)
    documents["CATEGORY_TOP_SHARE"] = (
        documents["CATEGORY_TOP_COUNT"] / documents["ITEM_COUNT"]
    ).astype(np.float32)
    documents["PRODUCT_TOP_SHARE"] = (
        documents["PRODUCT_TOP_COUNT"] / documents["ITEM_COUNT"]
    ).astype(np.float32)
    for column in ("PRODUCT_SET_HASH", "CATEGORY_SET_HASH"):
        documents[column] = (documents[column].fillna(0).astype("uint64") % 1000003).astype(np.int32)
    documents = documents.sort_values("SALESDOCUMENT").reset_index(drop=True)
    print(f"[phase] extraction rows={len(documents)} seconds={time.time() - started:.2f}")
    return documents


def attach_split(split, documents, target_col=None):
    columns = ["CREATIONTIMESTAMP", "SALESDOCUMENT"]
    if target_col is not None:
        columns.append(target_col)
    frame = split.df[columns].copy()
    frame["_row_index"] = np.arange(len(frame), dtype=np.int32)
    feature_columns = [column for column in documents.columns if column != "CREATIONTIMESTAMP"]
    frame = frame.merge(documents[feature_columns], on="SALESDOCUMENT", how="left", validate="one_to_one", sort=False)
    frame = frame.sort_values("_row_index").reset_index(drop=True)
    if frame["PAYERPARTY"].isna().any():
        raise RuntimeError("document feature join left missing payer rows")
    return frame
