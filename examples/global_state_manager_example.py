"""
Example demonstrating how to use the GlobalStateManager for shared state across Ray cluster.

This example shows:
1. Creating a named GlobalStateManager that persists across the cluster
2. Multiple TrainingManager instances sharing the same global state
3. Accessing the state manager from different parts of your code
4. Monitoring queue sizes and results across the cluster
"""

import asyncio
import ray
from tinkerbell.training.manager import TrainingManager, GlobalStateManager


async def example_basic_usage():
    """Basic usage: Create a training manager with global state."""
    print("=" * 60)
    print("Example 1: Basic Usage with Global State")
    print("=" * 60)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create a TrainingManager - it will automatically create or retrieve
    # a named global state manager that persists across the cluster
    manager1 = TrainingManager(
        clock_cycle=5.0,  # Process batches every 5 seconds
        state_manager_name="training_state_manager"  # Named actor
    )
    
    # Create another manager - it will retrieve the same state manager!
    manager2 = TrainingManager(
        clock_cycle=5.0,
        state_manager_name="training_state_manager"
    )
    
    # Both managers share the same global state
    print("✓ Created two TrainingManager instances sharing global state")
    
    # Get stats from the global state
    stats = await manager1.state_manager.get_stats.remote()
    print(f"Global state stats: {stats}")


async def example_explicit_state_manager():
    """Advanced usage: Explicitly create and pass a state manager."""
    print("\n" + "=" * 60)
    print("Example 2: Explicit State Manager Creation")
    print("=" * 60)
    
    # Explicitly create a named, detached state manager
    # "detached" means it survives even if the creating process dies
    state_manager = GlobalStateManager.options(
        name="my_custom_state_manager",
        lifetime="detached"
    ).remote()
    
    print("✓ Created detached global state manager: 'my_custom_state_manager'")
    
    # Pass it explicitly to multiple managers
    manager1 = TrainingManager(
        clock_cycle=5.0,
        global_state_manager=state_manager
    )
    
    manager2 = TrainingManager(
        clock_cycle=5.0,
        global_state_manager=state_manager
    )
    
    print("✓ Created two managers with explicit state manager")


async def example_retrieve_from_anywhere():
    """Show how to retrieve the state manager from anywhere in the cluster."""
    print("\n" + "=" * 60)
    print("Example 3: Retrieve State Manager from Anywhere")
    print("=" * 60)
    
    # First, ensure a state manager exists
    manager = TrainingManager(state_manager_name="training_state_manager")
    
    # Later, from anywhere in your Ray cluster, retrieve it by name
    try:
        state_manager = ray.get_actor("training_state_manager")
        print("✓ Retrieved global state manager from Ray cluster")
        
        # Now you can query it directly
        stats = await state_manager.get_stats.remote()
        print(f"Stats from anywhere in cluster: {stats}")
        
        # List all models
        models = await state_manager.list_models.remote()
        print(f"Registered models: {models}")
        
    except ValueError:
        print("State manager not found (not yet created)")


async def example_monitoring():
    """Example of monitoring the global state."""
    print("\n" + "=" * 60)
    print("Example 4: Monitoring Global State")
    print("=" * 60)
    
    manager = TrainingManager(state_manager_name="training_state_manager")
    
    # Get comprehensive stats
    stats = await manager.state_manager.get_stats.remote()
    print(f"\nGlobal State Statistics:")
    print(f"  Number of models: {stats['num_models']}")
    print(f"  Models: {stats['models']}")
    print(f"  Queue sizes: {stats['queue_sizes']}")
    print(f"  Number of results: {stats['num_results']}")
    
    # Check specific model queue size
    for model_name in stats['models']:
        queue_size = await manager.state_manager.get_queue_size.remote(model_name)
        print(f"  {model_name} queue size: {queue_size}")


async def example_multi_node_scenario():
    """
    Example simulating multiple nodes accessing the same global state.
    
    In a real multi-node Ray cluster, you would have:
    - Node 1: Creates actors, processes some requests
    - Node 2: Processes other requests, accesses same results
    - Node 3: Monitors the state
    
    All nodes share the same GlobalStateManager via Ray's actor system.
    """
    print("\n" + "=" * 60)
    print("Example 5: Multi-Node Scenario Simulation")
    print("=" * 60)
    
    # Simulate Node 1: Creates training actors
    print("\n[Node 1] Creating training actors...")
    manager_node1 = TrainingManager(
        clock_cycle=5.0,
        state_manager_name="multi_node_state"
    )
    
    # In a real scenario, you'd call:
    # await manager_node1.create_training_actors(
    #     world_size=4, 
    #     master_addr="...", 
    #     ...
    # )
    
    # Simulate Node 2: Another service using the same state
    print("[Node 2] Connecting to same state manager...")
    manager_node2 = TrainingManager(
        clock_cycle=5.0,
        state_manager_name="multi_node_state"
    )
    
    # Simulate Node 3: Monitoring service
    print("[Node 3] Monitoring service retrieving state...")
    state_manager = ray.get_actor("multi_node_state")
    stats = await state_manager.get_stats.remote()
    print(f"[Node 3] Monitoring stats: {stats}")
    
    print("\n✓ All nodes share the same global state via Ray actor")


async def example_cleanup():
    """Example of cleaning up the global state."""
    print("\n" + "=" * 60)
    print("Example 6: State Cleanup")
    print("=" * 60)
    
    manager = TrainingManager(state_manager_name="training_state_manager")
    
    # Remove a specific model's actor group
    removed = await manager.state_manager.remove_actor_group.remote("some_model")
    print(f"Removed actor group: {removed}")
    
    # Clear old results to free memory
    await manager.state_manager.clear_result.remote("some_request_id")
    print("Cleared specific result")
    
    # Get updated stats
    stats = await manager.state_manager.get_stats.remote()
    print(f"Updated stats: {stats}")


async def main():
    """Run all examples."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        await example_basic_usage()
        await example_explicit_state_manager()
        await example_retrieve_from_anywhere()
        await example_monitoring()
        await example_multi_node_scenario()
        await example_cleanup()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    finally:
        # Cleanup
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

