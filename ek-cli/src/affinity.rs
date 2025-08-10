use ek_base::config::CpuAffinityConfig;
use std::collections::HashSet;

/// CPU affinity operations trait for different operating systems
pub trait CpuAffinityOps {
    /// Apply CPU core affinity
    fn set_cpu_affinity(&self, cores: &[usize]) -> Result<(), String>;

    /// Apply NUMA node affinity
    fn set_numa_affinity(&self, numa_nodes: &[usize]) -> Result<(), String>;

    /// Get current CPU affinity mask
    #[allow(dead_code)]
    fn get_cpu_affinity(&self) -> Result<Vec<usize>, String>;

    /// Check if CPU affinity is supported on this platform
    fn is_cpu_affinity_supported(&self) -> bool;

    /// Check if NUMA affinity is supported on this platform
    fn is_numa_affinity_supported(&self) -> bool;

    /// Get platform-specific information
    fn get_platform_info(&self) -> String;

    /// Get the number of available CPU cores
    fn get_cpu_count(&self) -> usize;

    /// Get the number of available NUMA nodes
    fn get_numa_node_count(&self) -> usize;
}

/// Linux implementation of CPU affinity operations
#[cfg(target_os = "linux")]
pub struct LinuxCpuAffinityOps;

#[cfg(target_os = "linux")]
impl CpuAffinityOps for LinuxCpuAffinityOps {
    fn set_cpu_affinity(&self, cores: &[usize]) -> Result<(), String> {
        use libc::{CPU_SET, CPU_ZERO, cpu_set_t, sched_setaffinity};

        unsafe {
            let mut cpu_set: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut cpu_set);

            for &core in cores {
                if core >= libc::CPU_SETSIZE as usize {
                    return Err(format!("CPU core {core} exceeds CPU_SETSIZE limit"));
                }
                CPU_SET(core, &mut cpu_set);
            }

            let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpu_set);
            if result != 0 {
                let error = std::io::Error::last_os_error();
                return Err(format!("Failed to set CPU affinity: {error}"));
            }
        }

        Ok(())
    }

    fn set_numa_affinity(&self, numa_nodes: &[usize]) -> Result<(), String> {
        self.set_numa_cpu_affinity(numa_nodes)?;
        Ok(())
    }

    fn get_cpu_affinity(&self) -> Result<Vec<usize>, String> {
        use libc::{CPU_ISSET, cpu_set_t, sched_getaffinity};

        unsafe {
            let mut cpu_set: cpu_set_t = std::mem::zeroed();
            let result = sched_getaffinity(0, std::mem::size_of::<cpu_set_t>(), &mut cpu_set);

            if result != 0 {
                let error = std::io::Error::last_os_error();
                return Err(format!("Failed to get CPU affinity: {error}"));
            }

            let mut cores = Vec::new();
            for cpu in 0..libc::CPU_SETSIZE as usize {
                if CPU_ISSET(cpu, &cpu_set) {
                    cores.push(cpu);
                }
            }

            Ok(cores)
        }
    }

    fn is_cpu_affinity_supported(&self) -> bool {
        true
    }

    fn is_numa_affinity_supported(&self) -> bool {
        // Check if NUMA is available by checking if /sys/devices/system/node exists
        std::path::Path::new("/sys/devices/system/node").exists()
    }

    fn get_platform_info(&self) -> String {
        "Linux".to_string()
    }

    fn get_cpu_count(&self) -> usize {
        num_cpus::get()
    }

    fn get_numa_node_count(&self) -> usize {
        self.get_available_numa_nodes().len()
    }
}

#[cfg(target_os = "linux")]
impl LinuxCpuAffinityOps {
    /// Set CPU affinity for NUMA nodes
    fn set_numa_cpu_affinity(&self, numa_nodes: &[usize]) -> Result<(), String> {
        // Get CPU cores for the specified NUMA nodes
        let mut cpu_cores = Vec::new();

        for &node in numa_nodes {
            let node_cpus = self.get_numa_node_cpus(node)?;
            cpu_cores.extend(node_cpus);
        }

        if !cpu_cores.is_empty() {
            self.set_cpu_affinity(&cpu_cores)?;
        }

        Ok(())
    }

    /// Get CPU cores for a specific NUMA node
    fn get_numa_node_cpus(&self, node: usize) -> Result<Vec<usize>, String> {
        let cpulist_path = format!("/sys/devices/system/node/node{node}/cpulist");

        let content = std::fs::read_to_string(&cpulist_path)
            .map_err(|e| format!("Failed to read NUMA node {node} CPU list: {e}"))?;

        self.parse_cpu_list(content.trim())
    }

    /// Parse CPU list format like "0-3,8-11" or "0,2,4"
    fn parse_cpu_list(&self, cpu_list: &str) -> Result<Vec<usize>, String> {
        let mut cpus = Vec::new();

        for part in cpu_list.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            if part.contains('-') {
                // Range format like "0-3"
                let range_parts: Vec<&str> = part.split('-').collect();
                if range_parts.len() != 2 {
                    return Err(format!("Invalid CPU range format: {part}"));
                }

                let start: usize = range_parts[0]
                    .parse()
                    .map_err(|_| format!("Invalid CPU number: {}", range_parts[0]))?;
                let end: usize = range_parts[1]
                    .parse()
                    .map_err(|_| format!("Invalid CPU number: {}", range_parts[1]))?;

                if start > end {
                    return Err(format!("Invalid CPU range: {start} > {end}"));
                }

                for cpu in start..=end {
                    cpus.push(cpu);
                }
            } else {
                // Single CPU number
                let cpu: usize = part
                    .parse()
                    .map_err(|_| format!("Invalid CPU number: {part}"))?;
                cpus.push(cpu);
            }
        }

        cpus.sort_unstable();
        cpus.dedup();
        Ok(cpus)
    }

    /// Get available NUMA nodes
    fn get_available_numa_nodes(&self) -> Vec<usize> {
        let mut nodes = Vec::new();

        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if let Some(name_str) = name_str.strip_prefix("node")
                    && let Ok(node_num) = name_str.parse::<usize>()
                {
                    nodes.push(node_num);
                }
            }
        }

        nodes.sort_unstable();
        nodes
    }
}

/// Default implementation for unsupported platforms
#[expect(dead_code)]
pub struct DefaultCpuAffinityOps;

impl CpuAffinityOps for DefaultCpuAffinityOps {
    fn set_cpu_affinity(&self, _cores: &[usize]) -> Result<(), String> {
        log::warn!("CPU affinity is not supported on this platform");
        Ok(())
    }

    fn set_numa_affinity(&self, _numa_nodes: &[usize]) -> Result<(), String> {
        log::warn!("NUMA affinity is not supported on this platform");
        Ok(())
    }

    fn get_cpu_affinity(&self) -> Result<Vec<usize>, String> {
        log::warn!("Getting CPU affinity is not supported on this platform");
        Ok(vec![])
    }

    fn is_cpu_affinity_supported(&self) -> bool {
        false
    }

    fn is_numa_affinity_supported(&self) -> bool {
        false
    }

    fn get_platform_info(&self) -> String {
        "Unsupported Platform".to_string()
    }

    fn get_cpu_count(&self) -> usize {
        num_cpus::get()
    }

    fn get_numa_node_count(&self) -> usize {
        0
    }
}

/// Factory function to get the appropriate CPU affinity implementation
pub fn get_cpu_affinity_ops() -> Box<dyn CpuAffinityOps> {
    #[cfg(target_os = "linux")]
    {
        Box::new(LinuxCpuAffinityOps)
    }
    #[cfg(not(any(target_os = "linux")))]
    {
        Box::new(DefaultCpuAffinityOps)
    }
}

pub fn try_apply_cpu_affinity(settings: &ek_base::config::WorkerSettings) -> Result<(), String> {
    if let Some(advanced) = &settings.advanced {
        if let Some(cpu_affinity) = &advanced.cpu_affinity {
            validate_cpu_affinity_config(cpu_affinity)?;
            apply_cpu_affinity(cpu_affinity)?;
        } else {
            log::info!("No CPU affinity settings provided, skipping.");
        }
    } else {
        log::info!("No advanced settings provided, skipping CPU affinity configuration.");
    }
    Ok(())
}

/// Apply CPU affinity settings based on configuration
pub fn apply_cpu_affinity(config: &CpuAffinityConfig) -> Result<(), String> {
    let ops = get_cpu_affinity_ops();

    log::info!(
        "Applying CPU affinity settings on platform: {}",
        ops.get_platform_info()
    );
    log::info!(
        "Available CPUs: {}, Available NUMA nodes: {}",
        ops.get_cpu_count(),
        ops.get_numa_node_count()
    );

    // Apply CPU core affinity if specified and supported
    if let Some(cores) = &config.cores
        && !cores.is_empty()
    {
        if ops.is_cpu_affinity_supported() {
            ops.set_cpu_affinity(cores)?;
            log::info!("CPU affinity set to cores: {cores:?}");
        } else {
            log::warn!("CPU affinity requested but not supported on this platform");
        }
    }

    // Apply NUMA affinity if specified and supported
    if let Some(numa_nodes) = &config.numa_nodes
        && !numa_nodes.is_empty()
    {
        if ops.is_numa_affinity_supported() {
            ops.set_numa_affinity(numa_nodes)?;
            log::info!("NUMA affinity set to nodes: {numa_nodes:?}");
        } else {
            log::warn!("NUMA affinity requested but not supported on this platform");
        }
    }

    Ok(())
}

/// Validate CPU affinity configuration
pub fn validate_cpu_affinity_config(config: &CpuAffinityConfig) -> Result<(), String> {
    let ops = get_cpu_affinity_ops();
    let cpu_count = ops.get_cpu_count();
    let numa_count = ops.get_numa_node_count();

    // Validate CPU cores
    if let Some(cores) = &config.cores {
        if cores.is_empty() {
            return Err("CPU cores list cannot be empty".to_string());
        }

        let mut unique_cores = HashSet::new();
        for &core in cores {
            if !unique_cores.insert(core) {
                return Err(format!("Duplicate CPU core {core} in configuration"));
            }

            // Check against actual CPU count
            if core >= cpu_count {
                return Err(format!(
                    "CPU core {core} exceeds available CPU count {cpu_count} (cores are 0-indexed)"
                ));
            }
        }
    }

    // Validate NUMA nodes
    if let Some(numa_nodes) = &config.numa_nodes {
        if numa_nodes.is_empty() {
            return Err("NUMA nodes list cannot be empty".to_string());
        }

        let mut unique_nodes = HashSet::new();
        for &node in numa_nodes {
            if !unique_nodes.insert(node) {
                return Err(format!("Duplicate NUMA node {node} in configuration"));
            }

            // Check against actual NUMA node count
            if numa_count > 0 && node >= numa_count {
                return Err(format!(
                    "NUMA node {node} exceeds available NUMA node count {numa_count} (nodes are 0-indexed)"
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_affinity_ops_factory() {
        let ops = get_cpu_affinity_ops();

        // Should not panic and should return a valid implementation
        let platform = ops.get_platform_info();
        assert!(!platform.is_empty());

        // CPU count should be reasonable
        let cpu_count = ops.get_cpu_count();
        assert!(cpu_count > 0);
        assert!(cpu_count <= 1024); // Reasonable upper bound
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_parse_cpu_list() {
        let ops = LinuxCpuAffinityOps;

        // Test single CPU
        assert_eq!(ops.parse_cpu_list("0").unwrap(), vec![0]);

        // Test CPU range
        assert_eq!(ops.parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);

        // Test mixed format
        assert_eq!(ops.parse_cpu_list("0,2-4,7").unwrap(), vec![0, 2, 3, 4, 7]);

        // Test invalid formats
        assert!(ops.parse_cpu_list("0-").is_err());
        assert!(ops.parse_cpu_list("a-b").is_err());
        assert!(ops.parse_cpu_list("3-1").is_err()); // Invalid range
    }

    #[test]
    fn test_platform_support_detection() {
        let ops = get_cpu_affinity_ops();

        // Test that the methods don't panic
        let _cpu_support = ops.is_cpu_affinity_supported();
        let _numa_support = ops.is_numa_affinity_supported();
        let _platform = ops.get_platform_info();
        let _cpu_count = ops.get_cpu_count();
        let _numa_count = ops.get_numa_node_count();
    }

    #[test]
    fn test_set_cpu_affinity() {
        let ops = get_cpu_affinity_ops();

        // Test setting CPU affinity with valid cores
        let result = ops.set_cpu_affinity(&[0, 1, 4]);
        assert!(result.is_ok(), "Failed to set CPU affinity: {result:?}");
        // get real set CPU affinity
        let real_affinity = ops.get_cpu_affinity().unwrap();
        assert_eq!(real_affinity, vec![0, 1, 4]);

        // Test setting CPU affinity with invalid core
        let result = ops.set_cpu_affinity(&[9999]);
        assert!(result.is_err(), "Expected error for invalid CPU core");

        // Test setting CPU affinity with empty list
        let result = ops.set_cpu_affinity(&[]);
        assert!(result.is_err(), "Expected error for empty CPU core list");
    }

    #[test]
    fn test_set_numa_affinity() {
        let ops = get_cpu_affinity_ops();

        // Test setting NUMA affinity with valid nodes
        let result = ops.set_numa_affinity(&[0, 1]);
        assert!(result.is_ok(), "Failed to set NUMA affinity: {result:?}");

        // Test setting NUMA affinity with invalid node
        let result = ops.set_numa_affinity(&[9999]);
        assert!(result.is_err(), "Expected error for invalid NUMA node");
    }
}
