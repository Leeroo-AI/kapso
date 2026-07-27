FROM node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3 AS common

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        procps \
        python3 \
        python3-zstandard \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 kapso-provider \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /usr/sbin/nologin kapso-provider

COPY src/kapso /usr/lib/python3/dist-packages/kapso

RUN chmod --recursive a+rX /usr/lib/python3/dist-packages/kapso \
    && printf '%s\n' '#!/usr/bin/python3' \
        'from kapso.cross_run.launch.run_action_coding_agent_consumer import main' \
        'main()' > /usr/local/bin/kapso-coding-agent-consumer \
    && printf '%s\n' '#!/usr/bin/python3' \
        'from kapso.cross_run.launch.run_action_coding_agent_runtime import main' \
        'main()' > /usr/local/bin/kapso-run-action-coding-agent-provider-sandbox \
    && printf '%s\n' '#!/usr/bin/python3' \
        'from kapso.cross_run.launch.run_action_coding_agent_supervisor import main' \
        'main()' \
        > /usr/local/bin/kapso-coding-agent-supervisor \
    && chmod 0755 \
        /usr/local/bin/kapso-coding-agent-consumer \
        /usr/local/bin/kapso-coding-agent-supervisor \
        /usr/local/bin/kapso-run-action-coding-agent-provider-sandbox

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
