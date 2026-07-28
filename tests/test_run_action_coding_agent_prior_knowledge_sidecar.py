import os

import pytest

from kapso.cross_run.launch.run_action_atomic_publication import (
    open_run_action_anonymous_file,
)
from kapso.cross_run.launch.run_action_coding_agent_prior_knowledge_sidecar import (
    RunActionCodingAgentPriorKnowledgeSidecarError,
    _DescriptorAuditSink,
)


def test_descriptor_audit_sink_appends_only_complete_bounded_events(tmp_path):
    directory_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    audit_descriptor = open_run_action_anonymous_file(directory_descriptor, 0o600)
    os.fchmod(audit_descriptor, 0o600)
    sink = _DescriptorAuditSink(
        descriptor=audit_descriptor,
        maximum_bytes=6,
        owner_user_id=os.geteuid(),
        owner_group_id=os.getegid(),
    )

    sink(b"{}\n")
    sink(b"[]\n")

    assert os.pread(audit_descriptor, 7, 0) == b"{}\n[]\n"
    assert os.fstat(audit_descriptor).st_nlink == 0
    with pytest.raises(
        RunActionCodingAgentPriorKnowledgeSidecarError,
        match="unsafe or full",
    ):
        sink(b"0\n")

    os.close(audit_descriptor)
    os.close(directory_descriptor)
