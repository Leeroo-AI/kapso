# Nginx Reverse Proxy Setup

This directory contains scripts to set up nginx as a reverse proxy with SSL for the Leeroopedia services.

## Quick Start

```bash
# Make script executable
chmod +x setup.sh

# Run with defaults (test-leeroopedia.leeroo.com)
sudo ./setup.sh

# Or specify custom domain and email
sudo ./setup.sh --domain your-domain.com --email your-email@example.com
```

## Prerequisites

Before running the setup script:

1. **DNS Configuration**: Your domain must point to this server's IP address
2. **Firewall/Security Group**: Ports 80 and 443 must be open for inbound traffic
3. **Services Running**: Wiki service (port 8080) and Leeroopedia API (port 8091) must be running

## What the Script Does

1. Installs nginx and certbot (if not already installed)
2. Creates nginx configuration that:
   - Serves the wiki at the root path (`/`)
   - Serves the Leeroopedia API at `/api/`
3. Obtains a free SSL certificate from Let's Encrypt
4. Configures automatic HTTP → HTTPS redirect
5. Sets up automatic certificate renewal (via certbot timer)

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--domain` | `test-leeroopedia.leeroo.com` | Domain name for the site |
| `--email` | `admin@leeroo.com` | Email for Let's Encrypt notifications |
| `--wiki-port` | `8080` | Port where wiki service is running |
| `--api-port` | `8091` | Port where Leeroopedia API is running |

## URLs After Setup

| Service | URL |
|---------|-----|
| Wiki | `https://your-domain.com/` |
| Leeroopedia API | `https://your-domain.com/api/` |
| API Documentation | `https://your-domain.com/api/docs` |

## Managing the Setup

### Check SSL Certificate Status

```bash
sudo certbot certificates
```

### Check Auto-Renewal Timer

```bash
sudo systemctl list-timers | grep certbot
```

### Test Certificate Renewal

```bash
sudo certbot renew --dry-run
```

### View Nginx Logs

```bash
# Access logs
sudo tail -f /var/log/nginx/access.log

# Error logs
sudo tail -f /var/log/nginx/error.log
```

### Restart Nginx

```bash
sudo systemctl restart nginx
```

## Troubleshooting

### Certificate Request Fails

- Ensure DNS is properly configured (domain points to server IP)
- Ensure ports 80 and 443 are open in your firewall/security group
- Check if nginx is running: `sudo systemctl status nginx`

### 502 Bad Gateway

- Ensure the wiki and API services are running
- Check if services are listening on the expected ports:
  ```bash
  ss -tlnp | grep -E ':8080|:8091'
  ```

### SSL Certificate Expired

Certbot should auto-renew, but if it fails:

```bash
sudo certbot renew --force-renewal
sudo systemctl reload nginx
```
