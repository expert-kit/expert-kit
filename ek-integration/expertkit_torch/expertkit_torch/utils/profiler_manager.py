# profiler_manager.py
import time
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class Metric:
    """Class to encapsulate performance metrics"""
    name: str
    total_time: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, time_taken: float, tokens: int = 0):
        """Update metric with new data"""
        self.total_time += time_taken
        self.total_tokens += tokens
        self.call_count += 1
    
    @property
    def avg_time(self) -> float:
        """Calculate average time per call"""
        return self.total_time / self.call_count if self.call_count > 0 else 0
    
    @property
    def tps(self) -> float:
        """Calculate tokens per second"""
        return self.total_tokens / self.total_time if self.total_time > 0 else 0
    
    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """Convert metric to dictionary for reporting"""
        return {
            "name": self.name,
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "avg_time": self.avg_time,
            "tps": self.tps,
            **self.additional_data
        }


class ProfilerManager:
    """
    Manager for profiling model components and collecting performance metrics.
    Can track different phases (prefill/decode) and component-specific metrics.
    
    Can be used as a context manager:
    ```
    with ProfilerManager(detailed_profile=True) as profiler:
        profiler.wrap_model(model)
        # Run inference
        outputs = model.generate(...)
        # Report is automatically generated on exit
    ```
    """
    
    def __init__(self, auto_report: bool = True, batch_size: int = 1):
        """
        Initialize the profiler manager.
        
        Args:
            auto_report: Whether to automatically print report when the context exits
            batch_size: The batch size used for reporting per-prompt metrics
        """
        self.auto_report = auto_report
        self.batch_size = batch_size
        self.metrics = {}
        self.module_hooks = {}
        self.wrapped_models = []
        self.is_prefill = True  # Starts with prefill phase
        self.total_start_time = None
        self.phase_start_time = None
        self.prefill_tokens = 0
        self.decode_tokens = 0
    
    def __enter__(self):
        """Enter the runtime context for profiling"""
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and cleanup resources"""
        # Unwrap all models
        for model in self.wrapped_models:
            self.unwrap_model(model)
        
        # Generate report if auto_report is enabled
        if self.auto_report and not exc_type:  # Only report if no exception occurred
            self.print_report(self.batch_size)
        
        # Don't suppress exceptions
        return False
        
    def reset(self):
        """Reset profiler state to start a new inference"""
        self.is_prefill = True
        self.total_start_time = None
        self.phase_start_time = None
        self.prefill_tokens = 0
        self.decode_tokens = 0
        
        # Clear all metrics to start fresh
        self.metrics.clear()
        
    def register_metric(self, name: str) -> Metric:
        """Register a new metric"""
        self.metrics[name] = Metric(name=name)
        return self.metrics[name]
        
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        return self.metrics.get(name)
    
    def update_metric(self, name: str, time_taken: float, tokens: int = 0):
        """Update a metric (or create if it doesn't exist)"""
        if name not in self.metrics:
            self.register_metric(name)
        self.metrics[name].update(time_taken, tokens)
    
    def add_module_hook(self, module: nn.Module, name: str):
        """Add a profiling hook to a specific module"""
        original_forward = module.forward
        
        def profiled_forward(*args, **kwargs):
            # Record current phase
            phase = "prefill" if self.is_prefill else "decode"
            
            # Start timing
            start_time = time.time()
            outputs = original_forward(*args, **kwargs)
            end_time = time.time()
            
            # Update metric
            metric_name = f"{name}_{phase}"
            self.update_metric(metric_name, end_time - start_time)
            
            return outputs
        
        # Replace the forward method - ensure we're not double-wrapping
        if not hasattr(module.forward, '_profiled'):
            module.forward = profiled_forward
            module.forward._profiled = True  # Mark as profiled to avoid double-wrapping
            
            # Store original method for later restoration
            self.module_hooks[module] = original_forward
        
        return module
    
    def inject_hooks(self, model: nn.Module):
        """
        Find and add profiling hooks to attention and expert modules.
        More selective to avoid double-counting at different levels.
        """
        # Track which parent paths already have hooks
        hooked_prefixes = set()
        
        # First, identify all module paths by type
        all_modules = {}
        for name, module in model.named_modules():
            # Skip the root module
            if name == '':
                continue
                
            module_type = module.__class__.__name__
            if module_type not in all_modules:
                all_modules[module_type] = []
            all_modules[module_type].append((name, module))
        
        # Function to check if a module path is contained within any hooked prefix
        def is_submodule_of_hooked(name):
            parts = name.split('.')
            for i in range(1, len(parts)):
                prefix = '.'.join(parts[:i])
                if prefix in hooked_prefixes:
                    return True
            return False
        
        # Add hooks to attention layers - prioritize specific attention module types
        attention_modules = []
        for type_name in ['Attention', 'SelfAttention', 'MultiHeadAttention']:
            if type_name in all_modules:
                attention_modules.extend(all_modules[type_name])
        
        if not attention_modules:
            for name, module in model.named_modules():
                if "attn" in name.lower() and not is_submodule_of_hooked(name):
                    attention_modules.append((name, module))
        
        # Add hooks to attention modules
        for name, module in attention_modules:
            if not is_submodule_of_hooked(name):
                self.add_module_hook(module, "attention")
                hooked_prefixes.add(name)
        
        # Add hooks to MoE blocks - prioritize specific MoE module types
        moe_modules = []
        for type_name in ['MoeLayer', 'SparseMoeBlock', 'MoeBlock', 'Qwen3MoeSparseMoeBlock']:
            if type_name in all_modules:
                moe_modules.extend(all_modules[type_name])
        
        if not moe_modules:
            for name, module in model.named_modules():
                if (any(x in name.lower() for x in ["moe", "expert", "mlp"]) and 
                    not is_submodule_of_hooked(name)):
                    moe_modules.append((name, module))
        
        # Add hooks to MoE modules
        for name, module in moe_modules:
            if not is_submodule_of_hooked(name):
                self.add_module_hook(module, "expert")
                hooked_prefixes.add(name)
        
        return model
    
    def wrap_model(self, model: nn.Module):
        """
        Wrap model with profiling hooks - convenience method that chains 
        wrap_model_forward and inject_hooks
        """
        # Wrap the forward method
        model = self.wrap_model_forward(model)
        
        # Add component hooks
        self.inject_hooks(model)
            
        # Track this model
        if model not in self.wrapped_models:
            self.wrapped_models.append(model)
            
        return model
    
    def wrap_model_forward(self, model: nn.Module):
        """
        Wrap top-level model forward method to track prefill and decode phases.
        """
        original_forward = model.forward
        
        def profiled_forward(*args, **kwargs):
            # If this is a new sequence
            if self.total_start_time is None:
                self.total_start_time = time.time()
                self.phase_start_time = self.total_start_time
            
            # Calculate input batch size and tokens if available
            batch_size = 1  # Default
            input_tokens = 0
            
            if args and hasattr(args[0], 'shape'):
                if len(args[0].shape) >= 2:  # [batch_size, seq_len, ...]
                    batch_size = args[0].shape[0]
                input_tokens = args[0].numel()
            elif 'input_ids' in kwargs and hasattr(kwargs['input_ids'], 'shape'):
                if len(kwargs['input_ids'].shape) >= 2:
                    batch_size = kwargs['input_ids'].shape[0]
                input_tokens = kwargs['input_ids'].numel()
            
            # Call the original forward
            outputs = original_forward(*args, **kwargs)
            
            # Record timing based on the phase
            now = time.time()
            
            if self.is_prefill:
                # This is the first forward pass - prefill phase
                self.prefill_tokens = input_tokens
                elapsed = now - self.phase_start_time
                
                # Create a new metric for prefill if it doesn't exist
                if "prefill" not in self.metrics:
                    self.register_metric("prefill")
                # Otherwise, reset it instead of accumulating
                else:
                    self.metrics["prefill"].total_time = 0
                    self.metrics["prefill"].total_tokens = 0
                    self.metrics["prefill"].call_count = 0
                
                self.update_metric("prefill", elapsed, input_tokens)
                
                # Mark that we're now in decode phase
                self.is_prefill = False
                self.phase_start_time = now
            else:
                # For decode phase, each sequence gets one new token per forward pass
                # So the number of new tokens is equal to the batch size
                new_tokens = batch_size
                self.decode_tokens += new_tokens
                
                elapsed = now - self.phase_start_time
                self.update_metric("decode", elapsed, new_tokens)
                self.phase_start_time = now
            
            # Update total time as the sum of prefill and decode
            if "prefill" in self.metrics and "decode" in self.metrics:
                prefill_time = self.metrics["prefill"].total_time
                decode_time = self.metrics["decode"].total_time
                total_tokens = self.prefill_tokens + self.decode_tokens
                
                # Reset total_generation metric instead of accumulating
                if "total_generation" not in self.metrics:
                    self.register_metric("total_generation")
                else:
                    self.metrics["total_generation"].total_time = 0
                    self.metrics["total_generation"].total_tokens = 0
                    self.metrics["total_generation"].call_count = 0
                
                # Update with the sum of prefill and decode
                self.update_metric("total_generation", prefill_time + decode_time, total_tokens)
            
            return outputs
        
        # Store original for restoration
        model._original_forward = original_forward
        model.forward = profiled_forward
        
        # Track this model
        if model not in self.wrapped_models:
            self.wrapped_models.append(model)
        
        return model
    
    def unwrap_model(self, model: nn.Module):
        """Remove profiling hooks and restore original methods"""
        # Restore original forward method
        if hasattr(model, '_original_forward'):
            model.forward = model._original_forward
            delattr(model, '_original_forward')
        
        # Restore original module forward methods
        for module, original_forward in self.module_hooks.items():
            if hasattr(module, 'forward') and module.forward.__name__ == 'profiled_forward':
                module.forward = original_forward
        
        # Remove from tracked models
        if model in self.wrapped_models:
            self.wrapped_models.remove(model)
        
        return model
    
    def report(self) -> Dict[str, Dict[str, Any]]:
        """Generate a report of all collected metrics"""
        return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def print_report(self, batch_size: int = None):
        """Print a formatted performance report"""
        # Use instance-level setting if not explicitly specified
        if batch_size is None:
            batch_size = self.batch_size
            
        print(f"\n===== MoE Performance Report (Batch Size: {batch_size}) =====")
        
        # Get primary metrics
        total_metric = self.metrics.get("total_generation")
        prefill_metric = self.metrics.get("prefill")
        decode_metric = self.metrics.get("decode")
        
        if not total_metric or not prefill_metric or not decode_metric:
            print("No data collected. Run inference first.")
            return
        
        # Overall stats
        print(f"Overall:")
        print(f"  Total time: {total_metric.total_time:.3f}s")
        print(f"  Total tokens: {total_metric.total_tokens} ({prefill_metric.total_tokens} input + {decode_metric.total_tokens} output)")
        print(f"  Throughput: {total_metric.tps:.2f} tokens/second")
        if batch_size > 1:
            print(f"  Per-prompt throughput: {total_metric.tps/batch_size:.2f} tokens/s")
        
        # Phase stats
        print()
        print(f"Phase breakdown:")
        print(f"  Prefill: {prefill_metric.total_time:.3f}s ({prefill_metric.total_time/total_metric.total_time*100:.1f}%), {prefill_metric.tps:.2f} tokens/s")
        print(f"  Decode: {decode_metric.total_time:.3f}s ({decode_metric.total_time/total_metric.total_time*100:.1f}%), {decode_metric.tps:.2f} tokens/s")
        print(f"  Average latency per token: {decode_metric.total_time/decode_metric.total_tokens*1000:.2f}ms")
        
        # Group metrics by component and phase
        attention_metrics = {k: m for k, m in self.metrics.items() if "attention" in k.lower()}
        expert_metrics = {k: m for k, m in self.metrics.items() if "expert" in k.lower()}
        
        # Attention metrics
        attention_prefill_metrics = {k: m for k, m in attention_metrics.items() if "prefill" in k}
        attention_decode_metrics = {k: m for k, m in attention_metrics.items() if "decode" in k}
        
        attention_prefill_time = sum([m.total_time for m in attention_prefill_metrics.values()], 0)
        attention_decode_time = sum([m.total_time for m in attention_decode_metrics.values()], 0)
        attention_total_time = attention_prefill_time + attention_decode_time
        
        # Expert metrics
        expert_prefill_metrics = {k: m for k, m in expert_metrics.items() if "prefill" in k}
        expert_decode_metrics = {k: m for k, m in expert_metrics.items() if "decode" in k}
        
        expert_prefill_time = sum([m.total_time for m in expert_prefill_metrics.values()], 0)
        expert_decode_time = sum([m.total_time for m in expert_decode_metrics.values()], 0)
        expert_total_time = expert_prefill_time + expert_decode_time
        
        # Component breakdown for total time and each phase
        print()
        print("Component breakdown:")
        
        # Total (Prefill + Decode)
        if total_metric.total_time > 0:
            component_sum = attention_total_time + expert_total_time
            others_time = max(0.0, total_metric.total_time - component_sum)
            
            print(f"  Overall:")
            print(f"    Attention: {attention_total_time:.3f}s ({attention_total_time/total_metric.total_time*100:.1f}%)")
            print(f"    Expert: {expert_total_time:.3f}s ({expert_total_time/total_metric.total_time*100:.1f}%)")
            print(f"    Other: {others_time:.3f}s ({others_time/total_metric.total_time*100:.1f}%)")
        
        # Prefill phase
        if prefill_metric.total_time > 0:
            component_sum = attention_prefill_time + expert_prefill_time
            others_time = max(0.0, prefill_metric.total_time - component_sum)
            
            print(f"  Prefill:")
            print(f"    Attention: {attention_prefill_time:.3f}s ({attention_prefill_time/prefill_metric.total_time*100:.1f}%)")
            print(f"    Expert: {expert_prefill_time:.3f}s ({expert_prefill_time/prefill_metric.total_time*100:.1f}%)")
            print(f"    Other: {others_time:.3f}s ({others_time/prefill_metric.total_time*100:.1f}%)")
                
        # Decode phase
        if decode_metric.total_time > 0:
            component_sum = attention_decode_time + expert_decode_time
            others_time = max(0.0, decode_metric.total_time - component_sum)
            
            print(f"  Decode:")
            print(f"    Attention: {attention_decode_time:.3f}s ({attention_decode_time/decode_metric.total_time*100:.1f}%)")
            print(f"    Expert: {expert_decode_time:.3f}s ({expert_decode_time/decode_metric.total_time*100:.1f}%)")
            print(f"    Other: {others_time:.3f}s ({others_time/decode_metric.total_time*100:.1f}%)")
        
        # Average latency per component for token generation
        if decode_metric and decode_metric.total_tokens > 0:
            print()
            print(f"Average latency per token:")
            print(f"  Attention: {attention_decode_time/decode_metric.total_tokens*1000:.2f}ms")
            print(f"  Expert: {expert_decode_time/decode_metric.total_tokens*1000:.2f}ms")