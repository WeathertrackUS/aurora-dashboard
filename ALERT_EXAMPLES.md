# System Alert Examples

This file contains example configurations for the system alert banner.

## Basic Info Alert
```json
{
  "enabled": true,
  "message": "System maintenance scheduled for December 5th, 2025. Some features may be temporarily unavailable.",
  "type": "info",
  "dismissible": true
}
```

## Warning Alert (Data Issues)
```json
{
  "enabled": true,
  "message": "NOAA data feed experiencing intermittent issues. We are working on a fix. Updates may be delayed.",
  "type": "warning",
  "dismissible": true
}
```

## Error Alert (Service Down)
```json
{
  "enabled": true,
  "message": "Aurora map service is temporarily unavailable due to maintenance. Expected resolution: 2 hours.",
  "type": "error",
  "dismissible": false
}
```

## Success Alert (Issue Resolved)
```json
{
  "enabled": true,
  "message": "All systems operational! The data feed issue has been resolved. Thank you for your patience.",
  "type": "success",
  "dismissible": true
}
```

## Disable Alert
```json
{
  "enabled": false,
  "message": "",
  "type": "info",
  "dismissible": true
}
```

## Alert Types

- **info** - Blue banner for general information and announcements
- **warning** - Orange banner for non-critical issues and warnings
- **error** - Red banner for critical issues and service outages
- **success** - Green banner for positive updates and issue resolutions

## Tips

1. **Be concise** - Keep messages short and clear
2. **Include timeframes** - Let users know when issues will be resolved
3. **Use appropriate types** - Match the severity to the alert type
4. **Make critical alerts non-dismissible** - Set `"dismissible": false` for important messages
5. **Update regularly** - Change the alert when status changes
6. **Disable when resolved** - Set `"enabled": false` when the issue is fixed

## Automatic Updates

The dashboard checks for alert updates every 5 minutes. Changes to `alert_config.json` will be reflected automatically without restarting the server.
