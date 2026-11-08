# Markdown renderer

A paragraph with **bold**, *italic*, ~~struck~~, `inline code`, a [link](https://example.com/a/very/long/path/that/needs/wrapping) and an autolink <https://example.com>.
Soft breaks join lines; a hard break follows\
right here. Inline math $e^{i\pi} + 1 = 0$ and a footnote[^note].

## Lists

- First item with enough text that it has to wrap onto a second row at narrow widths
- Second item
  - Nested item
    - Deeper still
- [x] Finished task
- [ ] Open task

1. One
2. Two

   With a second paragraph.
10. Ten

## Quotes

> A plain quote that is also long enough to wrap when the terminal gets narrow.
>
> > Nested quote.

> [!WARNING]
> Alerts render with a coloured bar and a title.

## Code

```rust
fn main() {
	println!("tabs become spaces, and a long line like this one is hard-wrapped rather than reflowed");
}
```

    indented code block

## Tables

| Column | Aligned right | Centered |
|--------|--------------:|:--------:|
| short | 1 | x |
| a much longer cell that will need wrapping | 12345 | 你好 |

---

<div>raw html block</div>

![an image](image.png)

[^note]: Footnotes render at their definition.
