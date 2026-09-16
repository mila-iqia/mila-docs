# How SSH authentication works

When connecting to the cluster via SSH, authentication proceeds in
three steps:

```mermaid
sequenceDiagram
    participant C as Local machine
    participant S as Cluster (login node)
    participant M as MFA service

    C->>S:      Connection request (public key identity)
    S-->>C:     Key challenge
    C->>S:      Signed challenge response
        S->>M:  Request MFA verification
    M-->>C:     Prompt (TOTP/email code or Push notification)
    alt TOTP or email
        C->>S: 6-digit code
    else Push notification
        C->>S: Approve on phone
    end
    S-->>C:     Access granted
```

1. **Key exchange** — the local machine proves possession of the private key
   matching the public key stored on the cluster.
2. **MFA challenge** — once the key is accepted, the cluster prompts for the
   second factor.
3. **Validation** — for Push, tap "Approve" on the smartphone; for TOTP or
   email, type the code into the terminal prompt.