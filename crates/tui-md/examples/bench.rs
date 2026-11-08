use std::time::Instant;
fn main() {
    let code = include_str!("../src/layout.rs");
    let md = format!("```rust\n{code}\n```");
    let theme = tui_md::Theme::default();
    let t = Instant::now();
    tui_md::render_str("```rust\nfn a(){}\n```", &theme, 80);
    println!("first (cold load): {:?}", t.elapsed());
    let t = Instant::now();
    tui_md::render_str("```python\nx=1\n```", &theme, 80);
    println!("second language: {:?}", t.elapsed());
    let t = Instant::now();
    let rows = tui_md::render_str(&md, &theme, 80);
    println!(
        "{} lines of rust: {:?} ({} rows)",
        code.lines().count(),
        t.elapsed(),
        rows.len()
    );
    let plain = tui_md::Theme {
        syntax_theme: None,
        ..theme
    };
    let t = Instant::now();
    tui_md::render_str(&md, &plain, 80);
    println!("same, no highlight: {:?}", t.elapsed());
}
