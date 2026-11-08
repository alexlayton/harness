//! Which colour scheme highlights fenced code.

use std::fmt;

#[cfg(feature = "custom-themes")]
use std::sync::Arc;

/// Name of the theme used by [`SyntaxTheme::default`].
pub const DEFAULT_SYNTAX_THEME: &str = "base16-ocean.dark";

/// A syntax-highlighting colour scheme: either one of the bundled themes by
/// name, or a custom TextMate `.tmTheme` (the format editor themes ship in).
///
/// Unknown names are not an error at construction; code simply renders plain
/// (check with [`SyntaxTheme::is_available`] to validate user config).
/// Without the `highlight` feature every theme is unavailable.
#[derive(Clone)]
pub struct SyntaxTheme(Inner);

#[derive(Clone)]
enum Inner {
    Named(String),
    #[cfg(feature = "custom-themes")]
    Loaded(Arc<syntect::highlighting::Theme>),
}

/// A custom theme failed to load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxThemeError(String);

impl fmt::Display for SyntaxThemeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid syntax theme: {}", self.0)
    }
}

impl std::error::Error for SyntaxThemeError {}

impl SyntaxTheme {
    /// A bundled theme by name; see [`SyntaxTheme::builtin_names`].
    pub fn named(name: impl Into<String>) -> Self {
        Self(Inner::Named(name.into()))
    }

    /// Names of the bundled themes, sorted. Empty without the `highlight`
    /// feature.
    pub fn builtin_names() -> Vec<&'static str> {
        #[cfg(feature = "highlight")]
        {
            let mut names = crate::highlight::theme_names();
            names.sort_unstable();
            names
        }
        #[cfg(not(feature = "highlight"))]
        Vec::new()
    }

    /// Parse a custom theme from `.tmTheme` (XML plist) source.
    #[cfg(feature = "custom-themes")]
    pub fn from_tmtheme(source: &str) -> Result<Self, SyntaxThemeError> {
        let theme = syntect::highlighting::ThemeSet::load_from_reader(&mut std::io::Cursor::new(
            source.as_bytes(),
        ))
        .map_err(|error| SyntaxThemeError(error.to_string()))?;
        Ok(Self(Inner::Loaded(Arc::new(theme))))
    }

    /// Load a custom theme from a `.tmTheme` file.
    #[cfg(feature = "custom-themes")]
    pub fn from_tmtheme_file(path: impl AsRef<std::path::Path>) -> Result<Self, SyntaxThemeError> {
        let source = std::fs::read_to_string(path.as_ref())
            .map_err(|error| SyntaxThemeError(error.to_string()))?;
        Self::from_tmtheme(&source)
    }

    /// Whether this theme can actually highlight (the name exists and the
    /// `highlight` feature is on).
    pub fn is_available(&self) -> bool {
        match &self.0 {
            #[cfg(feature = "highlight")]
            Inner::Named(name) => crate::highlight::theme_names().contains(&name.as_str()),
            #[cfg(not(feature = "highlight"))]
            Inner::Named(_) => false,
            #[cfg(feature = "custom-themes")]
            Inner::Loaded(_) => true,
        }
    }

    #[cfg(feature = "highlight")]
    pub(crate) fn resolve(&self) -> Option<&syntect::highlighting::Theme> {
        match &self.0 {
            Inner::Named(name) => crate::highlight::builtin_theme(name),
            #[cfg(feature = "custom-themes")]
            Inner::Loaded(theme) => Some(theme),
        }
    }
}

impl Default for SyntaxTheme {
    fn default() -> Self {
        Self::named(DEFAULT_SYNTAX_THEME)
    }
}

impl From<&str> for SyntaxTheme {
    fn from(name: &str) -> Self {
        Self::named(name)
    }
}

impl PartialEq for SyntaxTheme {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Inner::Named(a), Inner::Named(b)) => a == b,
            #[cfg(feature = "custom-themes")]
            (Inner::Loaded(a), Inner::Loaded(b)) => Arc::ptr_eq(a, b),
            #[cfg(feature = "custom-themes")]
            _ => false,
        }
    }
}

impl Eq for SyntaxTheme {}

impl fmt::Debug for SyntaxTheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Inner::Named(name) => write!(f, "SyntaxTheme::named({name:?})"),
            #[cfg(feature = "custom-themes")]
            Inner::Loaded(theme) => write!(f, "SyntaxTheme::custom({:?})", theme.name),
        }
    }
}
