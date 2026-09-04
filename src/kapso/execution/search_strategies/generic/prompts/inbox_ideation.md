### Things only a person can provide
A candidate may depend on something this machine does not have and no
engineering supplies: a credential, access to a private resource, a file
that exists only on someone's computer. Do not design around such a gap —
no "honest zero", no placeholder result, no partial deliverable that
skips the part needing it, and no hunting this machine for credentials.
Specify the candidate as if the resource will be provided and name the
dependency in the solution on its own line, `Needs from the person:
<what, and where it should go>`. The implementation session asks the
person for it and is resumed once it is provided; that is cheaper than
any workaround.{{answered_block}}