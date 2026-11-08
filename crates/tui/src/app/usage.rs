//! Token and subscription-allowance formatting for the input trailer and `/usage`.

use super::*;

/// Compact token counts the way the context-length formatter does: `1_234`
/// renders as `1.2k`, `1_000_000` as `1M`.
pub(super) fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        let value = n as f64 / 1_000_000.0;
        if value.fract() == 0.0 {
            format!("{}M", value as u64)
        } else {
            format!("{value:.1}M")
        }
    } else if n >= 1_000 {
        let value = n as f64 / 1_000.0;
        if value.fract() == 0.0 {
            format!("{}k", value as u64)
        } else {
            format!("{value:.1}k")
        }
    } else {
        n.to_string()
    }
}

pub(super) fn format_subscription_usage(provider: &str, usage: &SubscriptionUsage) -> String {
    format_subscription_usage_at(provider, usage, OffsetDateTime::now_utc().unix_timestamp())
}

pub(super) fn format_subscription_usage_at(
    provider: &str,
    usage: &SubscriptionUsage,
    now_timestamp: i64,
) -> String {
    let mut output = format!("{provider} subscription usage");
    if let Some(plan) = usage.plan.as_deref().filter(|plan| !plan.is_empty()) {
        let _ = write!(output, " · {plan}");
    }
    let label_width = usage
        .windows
        .iter()
        .map(|window| window.label.chars().count())
        .max()
        .unwrap_or_default();
    for window in &usage.windows {
        let _ = write!(
            output,
            "\n{:<label_width$}  {:>3}% used",
            window.label, window.used_percent
        );
        if let Some(status) = window
            .status
            .as_deref()
            .filter(|status| !status.eq_ignore_ascii_case("ok"))
        {
            let _ = write!(output, " · {status}");
        }
        if let Some(reset) = format_subscription_reset(window, now_timestamp) {
            let _ = write!(output, " · {reset}");
        }
    }
    output
}

pub(super) fn format_subscription_reset(
    window: &SubscriptionUsageWindow,
    now_timestamp: i64,
) -> Option<String> {
    if let Some(seconds) = window.resets_after_seconds {
        return Some(format!("resets in {}", format_duration(seconds)));
    }
    let value = window.resets_at.as_deref()?;
    if let Some(reset_timestamp) = parse_reset_timestamp(value) {
        let remaining = reset_timestamp.saturating_sub(now_timestamp);
        return Some(if remaining <= 0 {
            "reset due".into()
        } else {
            format!("resets in {}", format_duration(remaining as u64))
        });
    }
    Some(format!("resets at {}", friendly_reset_time(value)))
}

pub(super) fn parse_reset_timestamp(value: &str) -> Option<i64> {
    value.parse::<i64>().ok().or_else(|| {
        OffsetDateTime::parse(value, &Rfc3339)
            .ok()
            .map(|timestamp| timestamp.unix_timestamp())
    })
}

pub(super) fn format_duration(seconds: u64) -> String {
    let units = [
        (7 * 86_400, "w"),
        (86_400, "d"),
        (3_600, "h"),
        (60, "m"),
        (1, "s"),
    ];
    let mut remaining = seconds;
    let mut parts = Vec::new();
    for (size, suffix) in units {
        let count = remaining / size;
        if count > 0 || (size == 1 && parts.is_empty()) {
            parts.push(format!("{count}{suffix}"));
            remaining %= size;
            if parts.len() == 2 {
                break;
            }
        }
    }
    parts.join(" ")
}

/// Make an unparseable RFC 3339-shaped timestamp compact as a fallback.
pub(super) fn friendly_reset_time(value: &str) -> String {
    let Some((date, time)) = value.split_once('T') else {
        return value.to_owned();
    };
    let time = time
        .strip_suffix('Z')
        .unwrap_or(time)
        .split('.')
        .next()
        .unwrap_or(time);
    format!("{date} {time}Z")
}
