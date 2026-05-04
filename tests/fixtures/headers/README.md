PSRDADA ASCII key-value headers used by M0+ plumbing smoke tests.

correlator_header_dsaX.txt
  Generic DSA-X correlator header. Vendored from
  /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt
  on lxd110h23 (date: 2026-05-04). Used by tools/dod/M0.sh::[M0:plumbing_junkdb]
  to feed dada_junkdb during the M0 buffer-plumbing smoke. Buffer-shape
  fields (HDR_SIZE, RESOLUTION, NCHAN, NPOL, NBIT, FILE_SIZE) are the only
  ones M0 cares about; obs-specific fields (UTC_START, SOURCE) are
  placeholder values.
