use super::vfs::{WorkspaceFs, split_relative};
use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec};
use async_trait::async_trait;
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tree_sitter::{Language, Node, Parser};

/// Source outline tool. Files are opened through the workspace capability.
pub struct OutlineTool {
    fs: Arc<WorkspaceFs>,
}

impl OutlineTool {
    pub fn new(fs: Arc<WorkspaceFs>) -> Self {
        Self { fs }
    }
}

fn language(path: &str) -> Option<Language> {
    let ext = PathBuf::from(path)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "swift" => tree_sitter_swift::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "c" | "h" => tree_sitter_c::LANGUAGE.into(),
        "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => tree_sitter_cpp::LANGUAGE.into(),
        "js" | "jsx" | "mjs" | "cjs" => tree_sitter_javascript::LANGUAGE.into(),
        "ts" | "mts" | "cts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        _ => return None,
    })
}

fn declaration(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "function_definition"
            | "function_declaration"
            | "function"
            | "method_definition"
            | "method_declaration"
            | "constructor_declaration"
            | "class_definition"
            | "class_declaration"
            | "class_specifier"
            | "struct_item"
            | "struct_specifier"
            | "type_declaration"
            | "type_spec"
            | "enum_item"
            | "enum_specifier"
            | "trait_item"
            | "impl_item"
            | "interface_declaration"
            | "interface_type"
            | "type_alias_declaration"
            | "lexical_declaration"
            | "variable_declaration"
            | "const_item"
            | "static_item"
            | "protocol_declaration"
            | "extension_declaration"
            | "struct_declaration"
            | "enum_declaration"
            | "property_declaration"
            | "init_declaration"
    )
}

const MAX_ENTRIES: usize = 500;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const MAX_SIGNATURE_BYTES: usize = 1200;
const OUTPUT_LIMIT_NOTICE: &str = "... outline truncated at output limit\n";
const ENTRY_LIMIT_NOTICE: &str = "... outline truncated after 500 declarations\n";

fn outline(
    source: &str,
    lang: Language,
    filter: Option<&str>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let mut parser = Parser::new();
    parser.set_language(&lang).map_err(|e| e.to_string())?;
    let tree = parser.parse(source, None).ok_or("parser failed")?;
    let mut lines = Vec::new();
    let mut entry_limit_hit = false;
    let filter = filter.map(str::to_lowercase);
    walk(
        tree.root_node(),
        source.as_bytes(),
        filter.as_deref(),
        &mut lines,
        &mut entry_limit_hit,
        cancel,
    );
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    if lines.is_empty() {
        return Ok("No matching declarations found".into());
    }
    let mut output = String::new();
    // Leave room for either notice so the complete result stays under the cap.
    let notice_space = OUTPUT_LIMIT_NOTICE.len().max(ENTRY_LIMIT_NOTICE.len());
    for line in lines {
        if output.len() + line.len() + 1 + notice_space > MAX_OUTPUT_BYTES {
            output.push_str(OUTPUT_LIMIT_NOTICE);
            return Ok(output);
        }
        output.push_str(&line);
        output.push('\n');
    }
    if entry_limit_hit {
        output.push_str(ENTRY_LIMIT_NOTICE);
    }
    Ok(output)
}

/// Pre-order walk over named nodes with an explicit stack. A recursive walk
/// overflowed the thread stack (aborting the whole process) on deeply nested
/// input such as a 1 MiB file of `(((…`; the explicit stack lives on the heap.
fn walk(
    root: Node<'_>,
    source: &[u8],
    filter: Option<&str>,
    out: &mut Vec<String>,
    entry_limit_hit: &mut bool,
    cancel: &CancellationToken,
) {
    let mut stack = vec![(root, 0usize)];
    let mut children = Vec::new();
    while let Some((node, depth)) = stack.pop() {
        if cancel.is_cancelled() || *entry_limit_hit {
            return;
        }
        let next_depth = visit(node, source, depth, filter, out, entry_limit_hit);
        if *entry_limit_hit {
            return;
        }
        let mut cursor = node.walk();
        children.clear();
        children.extend(node.named_children(&mut cursor));
        // Reverse so the first child is visited first.
        stack.extend(children.drain(..).rev().map(|child| (child, next_depth)));
    }
}

/// Emit one declaration entry for `node` when it is a declaration, and
/// return the depth for its children.
fn visit(
    node: Node<'_>,
    source: &[u8],
    depth: usize,
    filter: Option<&str>,
    out: &mut Vec<String>,
    entry_limit_hit: &mut bool,
) -> usize {
    let kind = node.kind();
    if !declaration(kind) {
        return depth;
    }
    // A signature ends at the first body node; keep multiline parameters intact.
    let body_start = (0..node.child_count())
        .filter_map(|i| node.child(i))
        .find(|child| {
            matches!(
                child.kind(),
                "block"
                    | "compound_statement"
                    | "statement_block"
                    | "class_body"
                    | "field_declaration_list"
                    | "declaration_list"
                    | "function_body"
                    | "type_body"
                    | "enum_class_body"
                    | "member_declaration_list"
            )
        })
        .map(|child| child.start_byte())
        .unwrap_or(node.end_byte());
    let end = body_start.min(node.end_byte());
    let header_end = end.min(node.start_byte().saturating_add(MAX_SIGNATURE_BYTES));
    let header = String::from_utf8_lossy(&source[node.start_byte()..header_end]);
    let header = header
        .trim()
        .trim_end_matches('{')
        .trim_end_matches(':')
        .trim();
    let mut header = header.split_whitespace().collect::<Vec<_>>().join(" ");
    if header_end < end {
        header.push_str(" … [signature truncated]");
    }
    if header.is_empty() {
        return depth;
    }
    // Match the displayed signature, not the body of the declaration.
    let matches_filter = filter.is_none_or(|query| header.to_lowercase().contains(query));
    if matches_filter {
        if out.len() == MAX_ENTRIES {
            *entry_limit_hit = true;
            return depth;
        }
        // Tree-sitter's end point is exclusive. If a node ends at the
        // start of a new line, that line is not part of the declaration.
        let end_row = node.end_position().row
            - usize::from(
                node.end_position().column == 0
                    && node.end_position().row > node.start_position().row,
            );
        out.push(format!(
            "{}line {}, count {} — {}: {}",
            "  ".repeat(depth.min(12)),
            node.start_position().row + 1,
            end_row + 1 - node.start_position().row,
            kind,
            header
        ));
    }
    depth + 1
}

#[async_trait]
impl Tool for OutlineTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "outline".into(),
                description: "Outline declarations and signatures in a workspace source file. Each entry gives a 1-indexed starting line and a line count, usable as read offset and limit. Long signatures and large outlines are marked when truncated. Supports Rust (.rs), Python (.py), Swift (.swift), Go (.go), C (.c, .h), C++ (.cc, .cpp, .cxx, .hpp, .hh, .hxx), JavaScript (.js, .jsx, .mjs, .cjs), and TypeScript (.ts, .tsx, .mts, .cts).".into(),
                parameters: json!({"type":"object","properties":{"path":{"type":"string","description":"Source file path relative to the workspace"},"filter":{"type":"string","description":"Optional case-insensitive text to match in declaration signatures"}},"required":["path"],"additionalProperties":false}),
            },
            prompt: ToolPrompt::new("Outline source files", ["Use outline first for large supported source files to locate declarations; use filter to narrow results and read with offset and limit for implementation details."]),
        }
    }
    fn concurrency(&self, _: &Value) -> Concurrency {
        Concurrency::ReadOnly
    }
    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let summary = format!("outline {path}");
        let fs = self.fs.clone();
        // Reading and parsing up to 1 MiB is CPU and blocking file work; keep
        // it off the async worker threads.
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let path = path.as_str();
            let lang = language(path).ok_or("unsupported source extension")?;
            let relative = if PathBuf::from(path).is_absolute() {
                PathBuf::from(path)
                    .strip_prefix(fs.root())
                    .map_err(|_| "outside workspace")?
                    .to_string_lossy()
                    .into_owned()
            } else {
                path.to_owned()
            };
            let components = split_relative(&relative).map_err(|e| e.to_string())?;
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            #[cfg(unix)]
            let source = {
                use std::io::Read;
                let fd = super::vfs::unix::open_file_relative(&fs, &components)
                    .map_err(|e| e.to_string())?;
                let file = std::fs::File::from(fd);
                let mut bytes = Vec::new();
                file.take(1024 * 1024 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                if bytes.len() > 1024 * 1024 {
                    return Err("file exceeds 1 MiB outline limit".into());
                }
                String::from_utf8(bytes).map_err(|_| "not UTF-8 text")?
            };
            #[cfg(not(unix))]
            let source: String = {
                let _ = components;
                return Err("workspace outlining is not supported on this platform".into());
            };
            let filter = args.get("filter").and_then(Value::as_str);
            outline(&source, lang, filter, &cancel)
        })
        .await
        .unwrap_or_else(|error| Err(format!("outline task failed: {error}")));
        match result {
            Ok(content) => ToolOutput {
                content,
                is_error: false,
                summary,
            },
            Err(content) => ToolOutput {
                content,
                is_error: true,
                summary,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_limit_is_reported_only_when_more_declarations_exist() {
        let source = (0..=MAX_ENTRIES)
            .map(|n| format!("fn function_{n}() {{}}\n"))
            .collect::<String>();
        let cancel = CancellationToken::new();
        let lang = tree_sitter_rust::LANGUAGE.into();
        let result = outline(&source, lang, None, &cancel).unwrap();
        assert_eq!(result.matches(" — function_item:").count(), MAX_ENTRIES);
        assert!(result.contains(ENTRY_LIMIT_NOTICE));
        assert!(!result.contains("function_500"));

        let filtered = outline(
            &source,
            tree_sitter_rust::LANGUAGE.into(),
            Some("FUNCTION_500"),
            &cancel,
        )
        .unwrap();
        assert!(filtered.contains("function_500"));
        assert!(!filtered.contains("function_499"));
        assert!(!filtered.contains(ENTRY_LIMIT_NOTICE));

        let without_extra = source.strip_suffix("fn function_500() {}\n").unwrap();
        let complete = outline(
            without_extra,
            tree_sitter_rust::LANGUAGE.into(),
            None,
            &cancel,
        )
        .unwrap();
        assert!(!complete.contains(ENTRY_LIMIT_NOTICE));
    }

    #[test]
    fn deeply_nested_source_does_not_overflow_the_stack() {
        // A recursive walk aborted the process on input like this; run on a
        // small stack so a regression fails here, not only in release builds.
        let depth = 200_000;
        let source = format!(
            "fn outer() {{}}\nconst X: () = {}{};\n",
            "(".repeat(depth),
            ")".repeat(depth)
        );
        let output = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let lang = tree_sitter_rust::LANGUAGE.into();
                outline(&source, lang, None, &CancellationToken::new())
            })
            .unwrap()
            .join()
            .expect("outline must not overflow the stack")
            .unwrap();
        assert!(output.contains("fn outer()"), "{output}");
    }

    #[test]
    fn long_signatures_show_the_cut_and_keep_the_read_range() {
        let params = (0..160)
            .map(|n| format!("arg_{n}: usize"))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!("fn long({params}) {{\n}}\n");
        let result = outline(
            &source,
            tree_sitter_rust::LANGUAGE.into(),
            None,
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(result.contains("line 1, count 2 — function_item: fn long("));
        assert!(result.contains("[signature truncated]"));
        assert!(result.len() <= MAX_OUTPUT_BYTES);
    }

    #[test]
    fn output_limit_has_a_notice_within_the_limit() {
        let params = (0..75)
            .map(|n| format!("arg_{n}: usize"))
            .collect::<Vec<_>>()
            .join(", ");
        let source = (0..100)
            .map(|n| format!("fn function_{n}({params}) {{}}\n"))
            .collect::<String>();
        let result = outline(
            &source,
            tree_sitter_rust::LANGUAGE.into(),
            None,
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(result.contains(OUTPUT_LIMIT_NOTICE));
        assert!(result.len() <= MAX_OUTPUT_BYTES);
    }
}
