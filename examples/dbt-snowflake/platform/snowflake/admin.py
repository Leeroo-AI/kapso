"""Shared by the setup scripts of this folder: the platform's config, the account admin's connection, and the
key pair of a service identity.

The admin's login is the one thing the config does not hold. `snowflake.admin.login_file` names a file with three
lines, read by path and never printed:

    SNOWFLAKE_USER=<the admin's login name>
    SNOWFLAKE_PASSWORD=<a password or a programmatic access token>
    SNOWFLAKE_ROLE=ACCOUNTADMIN

    from admin import CONFIG, SNOWFLAKE, connect, public_key_body, records
"""
import json
import pathlib

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CONFIG = json.loads((pathlib.Path(__file__).resolve().parent.parent / "config.json").read_text())
SNOWFLAKE = CONFIG["snowflake"]


def login():
    lines = pathlib.Path(SNOWFLAKE["admin"]["login_file"]).read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def connect(database=None):
    who = login()
    return snowflake.connector.connect(account=SNOWFLAKE["account"], user=who["SNOWFLAKE_USER"], password=who["SNOWFLAKE_PASSWORD"],
                                       role=who["SNOWFLAKE_ROLE"], warehouse=SNOWFLAKE["admin"]["warehouse"], database=database)


def records(cursor, statement, parameters=None):
    cursor.execute(statement, parameters)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def public_key_body(private_key_file):
    """The public half of an identity's key pair, as Snowflake stores it on a user. The pair is generated when
    its private key file is absent, in a directory only this user can read."""
    if not private_key_file.exists():
        private_key_file.parent.mkdir(parents=True, exist_ok=True)
        private_key_file.parent.chmod(0o700)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                       serialization.NoEncryption()))
        private_key_file.chmod(0o600)
        print(f"  generated {private_key_file}")
    key = serialization.load_pem_private_key(private_key_file.read_bytes(), password=None)
    pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return "".join(line for line in pem.splitlines() if "PUBLIC KEY" not in line)
