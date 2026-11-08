//! Render a Markdown file to stdout: `cargo run --example render -- [FILE] [WIDTH]`.
//! Defaults to the bundled sample and the terminal width.

use std::io::Read;

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let source = match args.next().as_deref() {
        None => include_str!("sample.md").to_owned(),
        Some("-") => {
            let mut source = String::new();
            std::io::stdin().read_to_string(&mut source)?;
            source
        }
        Some(path) => std::fs::read_to_string(path)?,
    };
    let width = args
        .next()
        .and_then(|w| w.parse().ok())
        .or_else(|| crossterm::terminal::size().ok().map(|(w, _)| w as usize))
        .unwrap_or(80);
    for row in tui_md::render_str(&source, &tui_md::Theme::default(), width) {
        println!("{}", row.to_ansi());
    }
    Ok(())
}
