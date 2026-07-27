FROM node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3 AS common

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libcap2-bin \
        procps \
        python3 \
        python3-zstandard \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 kapso-provider \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /usr/sbin/nologin kapso-provider \
    && test "$(readlink /sbin)" = 'usr/sbin' \
    && unlink /sbin \
    && mkdir --mode=0755 /sbin \
    && mkdir --mode=0555 --parents /kapso/credentials \
    && printf '{}\n' > /kapso/credentials/credentials \
    && chown 1000:1001 /kapso/credentials/credentials \
    && chmod 0440 /kapso/credentials/credentials

COPY src/kapso /usr/lib/python3/dist-packages/kapso

RUN chmod --recursive a+rX /usr/lib/python3/dist-packages/kapso \
    && cp /usr/bin/python3.11 /usr/local/bin/kapso-provider-python \
    && printf '%s\n' '#!/usr/bin/python3' \
        'from kapso.cross_run.launch.run_action_coding_agent_consumer import main' \
        'main()' > /usr/local/bin/kapso-coding-agent-consumer \
    && printf '%s\n' '#!/usr/bin/python3' \
        'from kapso.cross_run.launch.run_action_coding_agent_supervisor import main' \
        'main()' \
        > /usr/local/bin/kapso-coding-agent-supervisor \
    && chmod 0755 \
        /usr/local/bin/kapso-coding-agent-consumer \
        /usr/local/bin/kapso-provider-python \
        /usr/local/bin/kapso-coding-agent-supervisor \
    && setcap cap_kill,cap_setgid,cap_setpcap,cap_setuid=ep /usr/bin/python3.11 \
    && test -z "$(getcap /usr/local/bin/kapso-provider-python)" \
    && test "$(getcap /usr/bin/python3.11)" = \
        '/usr/bin/python3.11 cap_kill,cap_setgid,cap_setuid,cap_setpcap=ep'

FROM common AS offline-build

COPY tests/fixtures/coding_agent/offline_codex.py /usr/bin/codex
RUN chmod 0755 /usr/bin/codex

FROM scratch AS offline

COPY --from=offline-build / /
ENV LANG=C PATH=/usr/local/bin:/usr/bin:/bin

FROM common AS production-build

RUN npm install --global --ignore-scripts @openai/codex@0.144.1 \
    && ln -s /usr/local/bin/codex /usr/bin/codex \
    && test "$(/usr/bin/codex --version)" = 'codex-cli 0.144.1'

FROM scratch AS production

COPY --from=production-build / /
ENV LANG=C PATH=/usr/local/bin:/usr/bin:/bin
