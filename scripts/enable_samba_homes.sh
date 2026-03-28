#!/usr/bin/env bash
set -euo pipefail

SMB_CONF="/etc/samba/smb.conf"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash scripts/enable_samba_homes.sh"
  exit 1
fi

if [[ ! -f "$SMB_CONF" ]]; then
  echo "smb.conf not found: $SMB_CONF"
  exit 1
fi

backup="${SMB_CONF}.bak.$(date +%Y%m%d_%H%M%S)"
cp "$SMB_CONF" "$backup"
echo "Backup created: $backup"

# Remove an existing active [homes] block if present to avoid duplicates.
tmp="$(mktemp)"
awk '
  BEGIN {skip=0}
  /^\[homes\]$/ {skip=1; next}
  /^\[/ && skip==1 {skip=0}
  skip==0 {print}
' "$SMB_CONF" > "$tmp"
cat "$tmp" > "$SMB_CONF"
rm -f "$tmp"

cat >> "$SMB_CONF" <<'EOF'

[homes]
   comment = Home Directories
   browseable = no
   read only = no
   guest ok = no
   valid users = %S
   create mask = 0600
   directory mask = 0700
EOF

testparm -s >/dev/null
echo "smb.conf syntax OK"

systemctl restart smbd nmbd
systemctl enable smbd nmbd >/dev/null
echo "smbd/nmbd restarted and enabled"

echo
echo "Verify:"
echo "  testparm -s | grep -nE '^\\[homes\\]|browseable = no|read only = no|valid users = %S'"
echo
echo "Windows access:"
echo "  \\\\$(hostname -I | awk '{print $1}')\\<linux_username>"
echo "  Example: \\\\$(hostname -I | awk '{print $1}')\\shinsuke"
