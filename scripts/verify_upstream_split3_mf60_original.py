#!/usr/bin/env python3
"""Reopen the complete original MF60 outputs with original independent verifier."""
import upstream_split3_mf60_original as original
import verify_upstream_split3_next3 as verifier
verifier.batch=original.batch

if __name__=='__main__':verifier.verify('s3-mf60')
