from hashlib import blake2b

blake2b_digest_size = 12


def hashme(s: str):
    return blake2b(
        s.encode("utf-8"), digest_size=blake2b_digest_size
    ).hexdigest()
