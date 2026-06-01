use ek_base::error::{EKError, EKResult};

/// Invoke an external provisioner script to add new worker nodes.
///
/// The script is called as:
/// ```
/// <script> --count <N> --model <model> --instance <instance>
/// ```
/// A zero exit code indicates success; any non-zero exit code returns an error.
/// stdout/stderr from the script are forwarded to the controller log.
pub async fn provision_nodes(
    script: &str,
    count: u32,
    model: &str,
    instance: &str,
) -> EKResult<()> {
    log::info!(
        "provision_nodes: invoking '{}' --count {} --model {} --instance {}",
        script,
        count,
        model,
        instance
    );

    let script_owned = script.to_owned();
    let count_str = count.to_string();
    let model_owned = model.to_owned();
    let instance_owned = instance.to_owned();

    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new(&script_owned)
            .arg("--count")
            .arg(&count_str)
            .arg("--model")
            .arg(&model_owned)
            .arg("--instance")
            .arg(&instance_owned)
            .output()
    })
    .await
    .map_err(|e| EKError::RuntimeError(format!("spawn_blocking join error: {e}")))?
    .map_err(|e| EKError::RuntimeError(format!("Failed to launch provisioner '{script}': {e}")))?;

    // Forward stdout / stderr to logs for visibility
    if !output.stdout.is_empty() {
        log::info!(
            "provision_nodes stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
    if !output.stderr.is_empty() {
        log::warn!(
            "provision_nodes stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    if output.status.success() {
        log::info!("provision_nodes: completed successfully");
        Ok(())
    } else {
        Err(EKError::RuntimeError(format!(
            "Provisioner '{}' exited with status {}",
            script, output.status
        )))
    }
}
