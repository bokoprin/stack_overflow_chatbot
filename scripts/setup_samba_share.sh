#!/usr/bin/env bash
set -euo pipefail

SHARE_NAME="${SHARE_NAME:-samba_share}"
SHARE_PATH="${SHARE_PATH:-/home/shinsuke/samba_share}"
SHARE_USER="${SHARE_USER:-shinsuke}"
SMB_CONF="/etc/samba/smb.conf"

if [[ "$EUID" -ne 0 ]]; then
  echo "Run as root: sudo bash scripts/setup_samba_share.sh"
  exit 1
fi

if ! id -u "$SHARE_USER" >/dev/null 2>&1; then
  echo "User not found: $SHARE_USER"
  exit 1
fi

mkdir -p "$SHARE_PATH"
chown -R "$SHARE_USER:$SHARE_USER" "$SHARE_PATH"
chmod 2775 "$SHARE_PATH"

backup="$SMB_CONF.bak.$(date +%Y%m%d_%H%M%S)"
cp "$SMB_CONF" "$backup"
echo "Backup created: $backup"

if ! grep -q "^\[$SHARE_NAME\]$" "$SMB_CONF"; then
cat >> "$SMB_CONF" <<EOF

[$SHARE_NAME]
   path = $SHARE_PATH
   browseable = yes
   read only = no
   guest ok = no
   valid users = $SHARE_USER
   force user = $SHARE_USER
   create mask = 0664
   directory mask = 0775
EOF
  echo "Share section added: [$SHARE_NAME]"
else
  echo "Share section already exists: [$SHARE_NAME]"
fi

testparm -s >/dev/null
echo "smb.conf syntax OK"

systemctl restart smbd nmbd
systemctl enable smbd nmbd >/dev/null
echo "smbd/nmbd restarted and enabled"

if command -v ufw >/dev/null 2>&1; then
  ufw allow samba >/dev/null 2>&1 || true
fi

echo
echo "Next:"
echo "  sudo smbpasswd -a $SHARE_USER"
echo "Windows path:"
echo "  \\\\$(hostname -I | awk '{print $1}')\\$SHARE_NAME"
