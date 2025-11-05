//! TLS Certificate Management
//!
//! Provides flexible certificate management for different deployment scenarios:
//! 1. Existing certificates (production with reverse proxy)
//! 2. Self-signed certificates (development/internal use)
//! 3. ACME certificates (standalone with domain)
//! 4. HTTP-only mode (development/testing)

use anyhow::{Context, Result};
use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair, SanType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use tokio::fs;
use tracing::{info, warn};

/// TLS configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS (if false, server runs HTTP-only)
    pub enabled: bool,

    /// Certificate source configuration
    pub cert_source: CertificateSource,

    /// Bind address for HTTPS (if different from HTTP)
    pub https_bind_address: Option<String>,

    /// Force HTTPS redirects (redirect HTTP to HTTPS)
    pub force_https: bool,
}

/// Certificate source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CertificateSource {
    /// Use existing certificate files
    Existing {
        cert_path: PathBuf,
        key_path: PathBuf,
    },

    /// Generate self-signed certificate
    SelfSigned {
        /// Common name (usually hostname)
        common_name: String,
        /// Subject Alternative Names
        san_list: Vec<String>,
        /// Certificate storage directory
        cert_dir: PathBuf,
    },

    /// ACME certificate (Let's Encrypt, etc.)
    Acme {
        /// ACME directory URL (e.g., Let's Encrypt staging/prod)
        directory_url: String,
        /// Domain names for the certificate
        domains: Vec<String>,
        /// Contact email for ACME account
        contact_email: String,
        /// Certificate storage directory
        cert_dir: PathBuf,
        /// Cache directory for ACME account keys
        cache_dir: PathBuf,
    },
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cert_source: CertificateSource::SelfSigned {
                common_name: "localhost".to_string(),
                san_list: vec!["127.0.0.1".to_string(), "::1".to_string()],
                cert_dir: PathBuf::from("./certs"),
            },
            https_bind_address: None,
            force_https: false,
        }
    }
}

/// Certificate manager handles all certificate operations
pub struct CertificateManager {
    config: TlsConfig,
}

impl CertificateManager {
    /// Create a new certificate manager
    pub fn new(config: TlsConfig) -> Self {
        Self { config }
    }

    /// Get or create TLS server configuration (returns cert and key PEM bytes)
    pub async fn get_server_config(&self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        if !self.config.enabled {
            info!("TLS disabled, running HTTP-only");
            return Ok(None);
        }

        let (cert_pem, key_pem) = match &self.config.cert_source {
            CertificateSource::Existing {
                cert_path,
                key_path,
            } => {
                info!("Loading existing certificate from {:?}", cert_path);
                self.load_existing_certificate(cert_path, key_path).await?
            }

            CertificateSource::SelfSigned {
                common_name,
                san_list,
                cert_dir,
            } => {
                info!(
                    "Generating or loading self-signed certificate for {}",
                    common_name
                );
                self.get_self_signed_certificate(common_name, san_list, cert_dir)
                    .await?
            }

            CertificateSource::Acme {
                directory_url,
                domains,
                contact_email,
                cert_dir,
                cache_dir,
            } => {
                info!("Getting ACME certificate for domains: {:?}", domains);
                self.get_acme_certificate(
                    directory_url,
                    domains,
                    contact_email,
                    cert_dir,
                    cache_dir,
                )
                .await?
            }
        };

        Ok(Some((cert_pem, key_pem)))
    }

    /// Load existing certificate files
    async fn load_existing_certificate(
        &self,
        cert_path: &Path,
        key_path: &Path,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let cert_pem = fs::read(cert_path)
            .await
            .with_context(|| format!("Failed to read certificate file: {:?}", cert_path))?;

        let key_pem = fs::read(key_path)
            .await
            .with_context(|| format!("Failed to read private key file: {:?}", key_path))?;

        Ok((cert_pem, key_pem))
    }

    /// Get or generate self-signed certificate
    async fn get_self_signed_certificate(
        &self,
        common_name: &str,
        san_list: &[String],
        cert_dir: &Path,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let cert_path = cert_dir.join("cert.pem");
        let key_path = cert_dir.join("key.pem");

        // Check if certificate exists and is still valid
        if cert_path.exists() && key_path.exists() {
            if let Ok((cert_pem, key_pem)) =
                self.load_existing_certificate(&cert_path, &key_path).await
            {
                if self.is_certificate_valid(&cert_pem).unwrap_or(false) {
                    info!("Using existing valid self-signed certificate");
                    return Ok((cert_pem, key_pem));
                }
            }
        }

        // Generate new certificate
        info!("Generating new self-signed certificate");
        let (cert_pem, key_pem) = self.generate_self_signed_certificate(common_name, san_list)?;

        // Ensure certificate directory exists
        if let Some(parent) = cert_path.parent() {
            fs::create_dir_all(parent)
                .await
                .context("Failed to create certificate directory")?;
        }

        // Save certificate and key
        fs::write(&cert_path, &cert_pem)
            .await
            .context("Failed to write certificate file")?;

        fs::write(&key_path, &key_pem)
            .await
            .context("Failed to write private key file")?;

        info!("Self-signed certificate saved to {:?}", cert_dir);
        Ok((cert_pem, key_pem))
    }

    /// Generate a new self-signed certificate
    fn generate_self_signed_certificate(
        &self,
        common_name: &str,
        san_list: &[String],
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut params = CertificateParams::new(vec![common_name.to_string()])?;

        // Set up distinguished name
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, common_name);
        dn.push(DnType::OrganizationName, "Lightbulb");
        dn.push(DnType::OrganizationalUnitName, "AI Inference");
        params.distinguished_name = dn;

        // Add Subject Alternative Names
        params.subject_alt_names = san_list
            .iter()
            .map(|san| {
                if san.parse::<std::net::IpAddr>().is_ok() {
                    SanType::IpAddress(san.parse().unwrap())
                } else {
                    SanType::DnsName(san.as_str().try_into().unwrap())
                }
            })
            .collect();

        // Set validity period (1 year)
        let now = time::OffsetDateTime::now_utc();
        params.not_before = now;
        params.not_after = now + time::Duration::days(365);

        // Generate key pair and certificate
        let key_pair = KeyPair::generate()?;
        let cert = params.self_signed(&key_pair)?;

        let cert_pem = cert.pem().into_bytes();
        let key_pem = key_pair.serialize_pem().into_bytes();

        Ok((cert_pem, key_pem))
    }

    /// Get ACME certificate (Let's Encrypt, etc.)
    async fn get_acme_certificate(
        &self,
        _directory_url: &str,
        domains: &[String],
        _contact_email: &str,
        cert_dir: &Path,
        _cache_dir: &Path,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        info!(
            "Starting ACME certificate acquisition for domains: {:?}",
            domains
        );

        // Check if we have cached valid certificates
        let cert_path = cert_dir.join("acme_cert.pem");
        let key_path = cert_dir.join("acme_key.pem");

        if cert_path.exists() && key_path.exists() {
            if let Ok((cert_pem, key_pem)) =
                self.load_existing_certificate(&cert_path, &key_path).await
            {
                if self.is_certificate_valid(&cert_pem).unwrap_or(false) {
                    info!("Using existing valid ACME certificate");
                    return Ok((cert_pem, key_pem));
                }
            }
        }

        warn!(
            "ACME certificate acquisition requires HTTP-01 challenge setup.\n\
             To enable ACME:\n\
             1. Integrate AcmeChallengeHandler into your HTTP server\n\
             2. Add route for /.well-known/acme-challenge/\n\
             3. Ensure server is accessible on port 80\n\
             4. Complete instant-acme v0.8 integration\n\
             \n\
             For detailed implementation, see the AcmeChallengeHandler struct below.\n\
             Falling back to self-signed certificate for now."
        );

        // For now, fall back to self-signed
        let default_domain = "localhost".to_string();
        let common_name = domains.first().unwrap_or(&default_domain);

        warn!(
            "Falling back to self-signed certificate for {}",
            common_name
        );
        self.generate_self_signed_certificate(common_name, domains)
    }

    /// Check if certificate is still valid (not expired)
    fn is_certificate_valid(&self, cert_pem: &[u8]) -> Result<bool> {
        use x509_parser::prelude::*;

        // Convert PEM to DER for parsing
        let pem_str = std::str::from_utf8(cert_pem).context("Invalid UTF-8 in certificate PEM")?;
        let lines: Vec<&str> = pem_str
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let base64_str = lines.join("");

        use base64::Engine;
        let cert_der = base64::engine::general_purpose::STANDARD
            .decode(base64_str)
            .context("Failed to decode certificate base64")?;

        let (_, cert) = parse_x509_certificate(&cert_der)
            .map_err(|e| anyhow::anyhow!("Failed to parse certificate: {}", e))?;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs();

        let not_after = cert.validity.not_after.timestamp() as u64;

        // Consider valid if more than 30 days remaining
        Ok(not_after > now + (30 * 24 * 3600))
    }

    /// Get HTTPS bind address
    pub fn https_bind_address(&self, http_bind_address: &str) -> String {
        self.config.https_bind_address.clone().unwrap_or_else(|| {
            // Default: change port from HTTP to HTTPS
            if http_bind_address.contains(":8080") {
                http_bind_address.replace(":8080", ":8443")
            } else if http_bind_address.contains(":80") {
                http_bind_address.replace(":80", ":443")
            } else {
                format!(
                    "{}:8443",
                    http_bind_address.split(':').next().unwrap_or("0.0.0.0")
                )
            }
        })
    }

    /// Check if HTTPS should be enforced
    pub fn should_force_https(&self) -> bool {
        self.config.enabled && self.config.force_https
    }
}

/// HTTPS redirect middleware
pub async fn https_redirect_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::{IntoResponse, Redirect};

    // Check if request is already HTTPS
    if request.uri().scheme_str() == Some("https") {
        return next.run(request).await;
    }

    // Check for X-Forwarded-Proto header (reverse proxy)
    if let Some(proto) = request.headers().get("x-forwarded-proto") {
        if proto == "https" {
            return next.run(request).await;
        }
    }

    // Redirect to HTTPS
    let uri = request.uri();
    let https_uri = format!(
        "https://{}{}",
        uri.authority().map(|a| a.as_str()).unwrap_or("localhost"),
        uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/")
    );

    Redirect::permanent(&https_uri).into_response()
}

/// ACME HTTP-01 Challenge Handler
///
/// This structure stores challenge tokens and responses for ACME HTTP-01 validation.
/// It should be integrated with your HTTP server to respond to ACME challenges.
#[derive(Clone, Default)]
pub struct AcmeChallengeHandler {
    challenges: Arc<RwLock<HashMap<String, String>>>,
}

impl AcmeChallengeHandler {
    /// Create a new ACME challenge handler
    pub fn new() -> Self {
        Self {
            challenges: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a challenge response
    ///
    /// The key is the challenge token, and the value is the key authorization
    pub fn add_challenge(&self, token: String, key_authorization: String) {
        let mut challenges = self.challenges.write().unwrap();
        challenges.insert(token, key_authorization);
        info!("Added ACME challenge token");
    }

    /// Remove a challenge after validation
    pub fn remove_challenge(&self, token: &str) {
        let mut challenges = self.challenges.write().unwrap();
        challenges.remove(token);
        info!("Removed ACME challenge token");
    }

    /// Get challenge response for a given token
    pub fn get_challenge(&self, token: &str) -> Option<String> {
        let challenges = self.challenges.read().unwrap();
        challenges.get(token).cloned()
    }

    /// Middleware to handle ACME HTTP-01 challenges
    ///
    /// Should be added to your router to handle `/.well-known/acme-challenge/` requests
    pub async fn challenge_middleware(
        self,
        request: axum::extract::Request,
        next: axum::middleware::Next,
    ) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;

        // Check if this is an ACME challenge request
        let path = request.uri().path();
        if let Some(token) = path.strip_prefix("/.well-known/acme-challenge/") {
            if let Some(key_auth) = self.get_challenge(token) {
                return key_auth.into_response();
            }
            return (StatusCode::NOT_FOUND, "Challenge not found").into_response();
        }

        // Not an ACME challenge, continue to next middleware/handler
        next.run(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use x509_parser::prelude::*;

    /// Helper to decode PEM to DER for testing
    fn pem_to_der(pem: &[u8]) -> Vec<u8> {
        let pem_str = std::str::from_utf8(pem).unwrap();
        // Parse PEM and extract DER data (skip the BEGIN/END lines and decode base64)
        let lines: Vec<&str> = pem_str
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let base64_str = lines.join("");
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(base64_str)
            .unwrap()
    }

    #[tokio::test]
    async fn test_self_signed_certificate_generation() {
        let temp_dir = TempDir::new().unwrap();

        let config = TlsConfig {
            enabled: true,
            cert_source: CertificateSource::SelfSigned {
                common_name: "test.localhost".to_string(),
                san_list: vec!["127.0.0.1".to_string(), "test.localhost".to_string()],
                cert_dir: temp_dir.path().to_path_buf(),
            },
            https_bind_address: None,
            force_https: false,
        };

        let manager = CertificateManager::new(config);
        let result = manager.get_server_config().await.unwrap();

        assert!(result.is_some());
        let (cert_pem, key_pem) = result.unwrap();

        // Verify certificate and key were returned
        assert!(!cert_pem.is_empty());
        assert!(!key_pem.is_empty());

        // Verify files were created
        let cert_path = temp_dir.path().join("cert.pem");
        let key_path = temp_dir.path().join("key.pem");

        assert!(cert_path.exists());
        assert!(key_path.exists());

        // Parse and verify certificate details (convert PEM to DER first)
        let cert_der = pem_to_der(&cert_pem);
        let (_, cert) = parse_x509_certificate(&cert_der).unwrap();

        // Verify common name
        let common_name = cert.subject().iter_common_name().next().unwrap();
        assert_eq!(common_name.as_str().unwrap(), "test.localhost");

        // Verify validity period (should be ~365 days)
        let validity_duration =
            cert.validity().not_after.timestamp() - cert.validity().not_before.timestamp();
        let days = validity_duration / (24 * 3600);
        assert!(
            days >= 364 && days <= 366,
            "Certificate should be valid for ~365 days"
        );

        // Verify SAN extension exists
        let san_ext = cert
            .extensions()
            .iter()
            .find(|ext| ext.oid == x509_parser::oid_registry::OID_X509_EXT_SUBJECT_ALT_NAME);
        assert!(san_ext.is_some(), "Certificate should have SAN extension");
    }

    #[tokio::test]
    async fn test_certificate_caching() {
        let temp_dir = TempDir::new().unwrap();

        let config = TlsConfig {
            enabled: true,
            cert_source: CertificateSource::SelfSigned {
                common_name: "cache.test".to_string(),
                san_list: vec!["cache.test".to_string()],
                cert_dir: temp_dir.path().to_path_buf(),
            },
            https_bind_address: None,
            force_https: false,
        };

        let manager = CertificateManager::new(config.clone());

        // First generation
        let result1 = manager.get_server_config().await.unwrap();
        assert!(result1.is_some());
        let (cert1, _) = result1.unwrap();

        // Parse first certificate to get serial number
        let cert1_der = pem_to_der(&cert1);
        let (_, parsed_cert1) = parse_x509_certificate(&cert1_der).unwrap();
        let serial1 = parsed_cert1.serial.to_bytes_be();

        // Second call should use cached certificate
        let manager2 = CertificateManager::new(config);
        let result2 = manager2.get_server_config().await.unwrap();
        assert!(result2.is_some());
        let (cert2, _) = result2.unwrap();

        // Parse second certificate to get serial number
        let cert2_der = pem_to_der(&cert2);
        let (_, parsed_cert2) = parse_x509_certificate(&cert2_der).unwrap();
        let serial2 = parsed_cert2.serial.to_bytes_be();

        // Certificates should have same serial number (cached)
        assert_eq!(
            serial1, serial2,
            "Cached certificate should have same serial number"
        );
    }

    #[tokio::test]
    async fn test_disabled_tls() {
        let config = TlsConfig {
            enabled: false,
            ..Default::default()
        };

        let manager = CertificateManager::new(config);
        let server_config = manager.get_server_config().await.unwrap();

        assert!(server_config.is_none());
    }

    #[tokio::test]
    async fn test_certificate_validation() {
        let temp_dir = TempDir::new().unwrap();

        // Create a valid certificate
        let config = TlsConfig {
            enabled: true,
            cert_source: CertificateSource::SelfSigned {
                common_name: "validation.test".to_string(),
                san_list: vec!["validation.test".to_string()],
                cert_dir: temp_dir.path().to_path_buf(),
            },
            https_bind_address: None,
            force_https: false,
        };

        let manager = CertificateManager::new(config);
        let result = manager.get_server_config().await.unwrap();
        let (cert_pem, _) = result.unwrap();

        // Verify certificate is valid (pass PEM directly)
        assert!(manager.is_certificate_valid(&cert_pem).unwrap());
    }

    #[test]
    fn test_https_bind_address() {
        let config = TlsConfig {
            enabled: true,
            cert_source: CertificateSource::SelfSigned {
                common_name: "test".to_string(),
                san_list: vec![],
                cert_dir: PathBuf::from("/tmp"),
            },
            https_bind_address: None,
            force_https: false,
        };

        let manager = CertificateManager::new(config);

        // Test port 8080 -> 8443
        assert_eq!(manager.https_bind_address("0.0.0.0:8080"), "0.0.0.0:8443");

        // Test port 80 -> 443
        assert_eq!(manager.https_bind_address("0.0.0.0:80"), "0.0.0.0:443");

        // Test custom port
        assert_eq!(manager.https_bind_address("0.0.0.0:3000"), "0.0.0.0:8443");
    }

    #[test]
    fn test_should_force_https() {
        let config_no_force = TlsConfig {
            enabled: true,
            cert_source: CertificateSource::SelfSigned {
                common_name: "test".to_string(),
                san_list: vec![],
                cert_dir: PathBuf::from("/tmp"),
            },
            https_bind_address: None,
            force_https: false,
        };

        let manager = CertificateManager::new(config_no_force);
        assert!(!manager.should_force_https());

        let config_force = TlsConfig {
            enabled: true,
            force_https: true,
            ..Default::default()
        };

        let manager_force = CertificateManager::new(config_force);
        assert!(manager_force.should_force_https());
    }

    // Middleware tests
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, header},
        middleware::{self, Next},
        response::Response,
        routing::get,
    };
    use tower::ServiceExt;

    async fn handler() -> &'static str {
        "OK"
    }

    #[tokio::test]
    async fn test_redirect_middleware_http_to_https() {
        let app = Router::new()
            .route("/test", get(handler))
            .layer(middleware::from_fn(super::https_redirect_middleware));

        let request = Request::builder()
            .uri("http://example.com/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(
            response.headers().get(header::LOCATION).unwrap(),
            "https://example.com/test"
        );
    }

    #[tokio::test]
    async fn test_redirect_middleware_https_passthrough() {
        let app = Router::new()
            .route("/test", get(handler))
            .layer(middleware::from_fn(super::https_redirect_middleware));

        let request = Request::builder()
            .uri("https://example.com/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_redirect_middleware_x_forwarded_proto() {
        let app = Router::new()
            .route("/test", get(handler))
            .layer(middleware::from_fn(super::https_redirect_middleware));

        let request = Request::builder()
            .uri("http://example.com/test")
            .header("x-forwarded-proto", "https")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Should pass through because X-Forwarded-Proto indicates HTTPS
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_redirect_middleware_preserves_path_and_query() {
        let app = Router::new()
            .route("/api/endpoint", get(handler))
            .layer(middleware::from_fn(super::https_redirect_middleware));

        let request = Request::builder()
            .uri("http://api.example.com/api/endpoint?key=value&foo=bar")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        let location = response.headers().get(header::LOCATION).unwrap();
        assert_eq!(
            location.to_str().unwrap(),
            "https://api.example.com/api/endpoint?key=value&foo=bar"
        );
    }
}
