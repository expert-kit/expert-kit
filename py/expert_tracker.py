import json
import sqlite3
import torch
from datetime import datetime
from typing import Optional, List, Dict

# TODO: connection will be create each time a query is made, this is not efficient
# Maybe a connection pool should be used in future version
class ExpertTrackerBitmap:
    """
    Utility class for tracking expert activations in Mixture of Experts (MoE) models.
    This class uses SQLite3 to store and manage data about inference requests,
    generated tokens, and expert activations across model layers.
    """

    def __init__(self, db_path, num_layers=64, num_experts=256, experts_per_token=8):
        """
        Initialize the MoE tracker with database configuration.

        Args:
            db_path (str): Path to the SQLite database file
            num_layers (int): Number of MoE layers in the model
            num_experts (int): Number of experts per layer
            experts_per_token (int): Number of experts activated per layer during inference
        """
        self.db_path = db_path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        # Initialize the database
        self._init_db()

    def _init_db(self):
        """
        Initialize the SQLite database with required tables if they don't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create requests table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_text TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT NOT NULL
        )
        ''')

        # Create generations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS generations (
            generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            token_position INTEGER NOT NULL,
            token TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (request_id) REFERENCES requests (request_id)
        )
        ''')

        # Create expert activations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS expert_activations (
            activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            experts_bitmap BLOB NOT NULL,
            FOREIGN KEY (generation_id) REFERENCES generations (generation_id)
        )
        ''')

        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_generations_request_id ON generations (request_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_activations_generation_id ON expert_activations (generation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_domain ON requests (domain)')

        conn.commit()
        conn.close()

    def record_request(self, request_text: str, domain: Optional[str] = None) -> Optional[int]:
        """
        Record a new inference request.

        Args:
            request_text (str): The text of the request/prompt
            domain (str, optional): Domain/category of the request

        Returns:
            int: The ID of the recorded request
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        cursor.execute(
            'INSERT INTO requests (request_text, domain, timestamp) VALUES (?, ?, ?)',
            (request_text, domain, timestamp)
        )

        request_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return request_id

    def record_generation(
        self,
        request_id: int,
        token_position: int,
        expert_activations: List[List[int]],
        token: str,
    ) -> None:
        """
        Record a token generation with its expert activations across layers.

        Args:
            request_id (int): ID of the related request
            token_position (int): Position of the token in the generated sequence
            expert_activations (list): List of expert activations per layer.
                                      Each item should be a list of expert indices that were activated.
                                      Length should match self.num_layers.
            token (str): The generated token
        """
        if len(expert_activations) != self.num_layers:
            raise ValueError(f"Expected expert activations for {self.num_layers} layers, got {len(expert_activations)}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Record the token generation
        timestamp = datetime.now().isoformat()
        cursor.execute(
            'INSERT INTO generations (request_id, token_position, token, timestamp) VALUES (?, ?, ?, ?)',
            (request_id, token_position, token, timestamp)
        )

        generation_id = cursor.lastrowid

        # Record expert activations for each layer
        for layer, experts in enumerate(expert_activations):
            # Create bitmap of activated experts
            bitmap = self._create_expert_bitmap(experts)

            cursor.execute(
                'INSERT INTO expert_activations (generation_id, layer, experts_bitmap) VALUES (?, ?, ?)',
                (generation_id, layer, bitmap)
            )

        conn.commit()
        conn.close()

    def _create_expert_bitmap(self, activated_experts: List[int]) -> bytes:
        """
        Create a bitmap representing activated experts.

        Args:
            activated_experts (list): List of indices of activated experts

        Returns:
            bytes: Binary bitmap where each bit represents an expert
        """
        # Calculate number of bytes needed (ceil(num_experts / 8))
        num_bytes = (self.num_experts + 7) // 8

        # Create a bitmap initialized with zeros
        bitmap = bytearray(num_bytes)

        # Set bits for activated experts
        for expert_idx in activated_experts:
            if not (0 <= expert_idx < self.num_experts):
                raise ValueError(f"Expert index {expert_idx} out of range [0, {self.num_experts-1}]")

            byte_index = expert_idx // 8
            bit_position = expert_idx % 8
            bitmap[byte_index] |= (1 << bit_position)

        return bytes(bitmap)

    def _decode_expert_bitmap(self, bitmap: bytes) -> List[int]:
        """
        Decode a bitmap to get the list of activated experts.

        Args:
            bitmap (bytes): Binary bitmap where each bit represents an expert

        Returns:
            list: List of indices of activated experts
        """
        activated_experts = []

        for byte_index, byte_value in enumerate(bitmap):
            for bit_position in range(8):
                if byte_value & (1 << bit_position):
                    expert_idx = byte_index * 8 + bit_position
                    if expert_idx < self.num_experts:
                        activated_experts.append(expert_idx)

        return activated_experts

    def get_requests_by_domain(self, domain: Optional[str] = None) -> List[Dict]:
        """
        Get all requests, optionally filtered by domain.

        Args:
            domain (str, optional): Domain to filter by

        Returns:
            list: List of request records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if domain:
            cursor.execute('SELECT * FROM requests WHERE domain = ? ORDER BY timestamp', (domain,))
        else:
            cursor.execute('SELECT * FROM requests ORDER BY timestamp')

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_generations(self, request_id: int) -> List[Dict]:
        """
        Get all token generations for a specific request.

        Args:
            request_id (int): ID of the request

        Returns:
            list: List of generation records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM generations WHERE request_id = ? ORDER BY token_position',
            (request_id,)
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_expert_activations(self, generation_id: int) -> Dict[int, List[int]]:
        """
        Get expert activations for a specific token generation.

        Args:
            generation_id (int): ID of the generation

        Returns:
            dict: Dictionary mapping layer numbers to lists of activated experts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT layer, experts_bitmap FROM expert_activations WHERE generation_id = ? ORDER BY layer',
            (generation_id,)
        )

        activations = {}
        for layer, bitmap in cursor.fetchall():
            activated_experts = self._decode_expert_bitmap(bitmap)
            activations[layer] = activated_experts

        conn.close()

        return activations

    def get_expert_activation_stats_by_domain(self, domain: Optional[str] = None) -> Dict:
        """
        Get basic statistics about expert activations by domain.

        Args:
            domain (str, optional): Domain to filter by

        Returns:
            dict: Dictionary with activation statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
        SELECT ea.layer, ea.experts_bitmap
        FROM expert_activations ea
        JOIN generations g ON ea.generation_id = g.generation_id
        JOIN requests r ON g.request_id = r.request_id
        '''

        params = []
        if domain:
            query += ' WHERE r.domain = ?'
            params.append(domain)

        cursor.execute(query, params)

        # Initialize counters for each expert in each layer
        activation_counts = {layer: [0] * self.num_experts for layer in range(self.num_layers)}

        # Count activations
        for layer, bitmap in cursor.fetchall():
            activated_experts = self._decode_expert_bitmap(bitmap)
            for expert in activated_experts:
                activation_counts[layer][expert] += 1

        conn.close()

        return activation_counts

    def batch_record_generations(self, request_id: int, tokens: List[str], all_expert_activations: List[List[List[int]]]) -> None:
        """
        Record multiple token generations with their expert activations in a single transaction.
        This is more efficient than calling record_generation multiple times.

        Args:
            request_id (int): ID of the related request
            tokens (list): List of generated tokens
            all_expert_activations (list): List of expert activations for each token.
                                          Each item is a list of expert activations per layer.
        """
        if len(tokens) != len(all_expert_activations):
            raise ValueError(
                f"Number of tokens ({len(tokens)}) must match number of expert activation lists ({len(all_expert_activations)})")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        conn.execute("BEGIN TRANSACTION")
        timestamp = datetime.now().isoformat()

        for token_position, (token, expert_activations) in enumerate(zip(tokens, all_expert_activations)):
            if len(expert_activations) != self.num_layers:
                raise ValueError(
                    f"Expected expert activations for {self.num_layers} layers, got {len(expert_activations)}")

            # Record the token generation
            cursor.execute(
                'INSERT INTO generations (request_id, token_position, token, timestamp) VALUES (?, ?, ?, ?)',
                (request_id, token_position, token, timestamp)
            )

            generation_id = cursor.lastrowid

            # Record expert activations for each layer
            for layer, experts in enumerate(expert_activations):
                bitmap = self._create_expert_bitmap(experts)
                cursor.execute(
                    'INSERT INTO expert_activations (generation_id, layer, experts_bitmap) VALUES (?, ?, ?)',
                    (generation_id, layer, bitmap)
                )

        conn.commit()
        conn.close()

    def close(self):
        """
        Ensure any open database connections are properly closed.
        """
        pass


class ExpertTrackerReaderble:
    """
    Utility class for tracking expert activations in Mixture of Experts (MoE) models.
    This class uses SQLite3 to store and manage data about inference requests,
    generated tokens, and expert activations across model layers.

    This version uses human-readable formats for storing expert activations.
    """

    def __init__(self, db_path, num_layers=61, num_experts=256, experts_per_token=8):
        """
        Initialize the MoE tracker with database configuration.

        Args:
            db_path (str): Path to the SQLite database file
            num_layers (int): Number of MoE layers in the model
            num_experts (int): Number of experts per layer
            experts_per_token (int): Number of experts activated per layer during inference
        """
        self.db_path = db_path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        # Initialize the database
        self._init_db()

    def _init_db(self):
        """
        Initialize the SQLite database with required tables if they don't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create requests table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_text TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT NOT NULL
        )
        ''')

        # Create generations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS generations (
            generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            token_position INTEGER NOT NULL,
            token TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (request_id) REFERENCES requests (request_id)
        )
        ''')

        # Create expert activations table - using TEXT instead of BLOB for readability
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS expert_activations (
            activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            experts_list TEXT NOT NULL,
            FOREIGN KEY (generation_id) REFERENCES generations (generation_id)
        )
        ''')

        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_generations_request_id ON generations (request_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_activations_generation_id ON expert_activations (generation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_domain ON requests (domain)')

        conn.commit()
        conn.close()

    def record_request(self, request_text, domain=None):
        """
        Record a new inference request.

        Args:
            request_text (str): The text of the request/prompt
            domain (str, optional): Domain/category of the request

        Returns:
            int: The ID of the recorded request
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        cursor.execute(
            'INSERT INTO requests (request_text, domain, timestamp) VALUES (?, ?, ?)',
            (request_text, domain, timestamp)
        )

        request_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return request_id

    def get_requests_by_domain(self, domain: Optional[str] = None) -> List[Dict]:
        """
        Get all requests, optionally filtered by domain.

        Args:
            domain (str, optional): Domain to filter by

        Returns:
            list: List of request records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if domain:
            cursor.execute('SELECT * FROM requests WHERE domain = ? ORDER BY timestamp', (domain,))
        else:
            cursor.execute('SELECT * FROM requests ORDER BY timestamp')

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def record_generation(
        self,
        request_id,
        token_position,
        expert_activations,
        token,
    ):
        """
        Record a token generation with its expert activations across layers.

        Args:
            request_id (int): ID of the related request
            token_position (int): Position of the token in the generated sequence
            expert_activations (list): List of expert activations per layer.
                                      Each item should be a list of expert indices that were activated.
                                      Length should match self.num_layers.
            token (str): The generated token
        """
        if len(expert_activations) != self.num_layers:
            raise ValueError(f"Expected expert activations for {self.num_layers} layers, got {len(expert_activations)}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Record the token generation
        timestamp = datetime.now().isoformat()
        cursor.execute(
            'INSERT INTO generations (request_id, token_position, token, timestamp) VALUES (?, ?, ?, ?)',
            (request_id, token_position, token, timestamp)
        )

        generation_id = cursor.lastrowid

        # Record expert activations for each layer
        for layer, experts in enumerate(expert_activations):
            # Validate expert indices
            for expert_idx in experts:
                if not (0 <= expert_idx < self.num_experts):
                    raise ValueError(f"Expert index {expert_idx} out of range [0, {self.num_experts-1}]")

            # Store experts as a JSON list for human readability
            if isinstance(experts, torch.Tensor):
                experts = experts.tolist()
            experts_json = json.dumps(sorted(experts))

            cursor.execute(
                'INSERT INTO expert_activations (generation_id, layer, experts_list) VALUES (?, ?, ?)',
                (generation_id, layer, experts_json)
            )

        conn.commit()
        conn.close()

    def get_requests(self, domain=None):
        """
        Get all requests, optionally filtered by domain.

        Args:
            domain (str, optional): Domain to filter by

        Returns:
            list: List of request records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if domain:
            cursor.execute('SELECT * FROM requests WHERE domain = ? ORDER BY timestamp', (domain,))
        else:
            cursor.execute('SELECT * FROM requests ORDER BY timestamp')

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_generations(self, request_id):
        """
        Get all token generations for a specific request.

        Args:
            request_id (int): ID of the request

        Returns:
            list: List of generation records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM generations WHERE request_id = ? ORDER BY token_position',
            (request_id,)
        )

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_expert_activations(self, generation_id):
        """
        Get expert activations for a specific token generation.

        Args:
            generation_id (int): ID of the generation

        Returns:
            dict: Dictionary mapping layer numbers to lists of activated experts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT layer, experts_list FROM expert_activations WHERE generation_id = ? ORDER BY layer',
            (generation_id,)
        )

        activations = {}
        for layer, experts_json in cursor.fetchall():
            activated_experts = json.loads(experts_json)
            activations[layer] = activated_experts

        conn.close()

        return activations

    def get_expert_activation_stats_by_domain(self, domain=None):
        """
        Get basic statistics about expert activations by domain.

        Args:
            domain (str, optional): Domain to filter by

        Returns:
            dict: Dictionary with activation statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
        SELECT ea.layer, ea.experts_list
        FROM expert_activations ea
        JOIN generations g ON ea.generation_id = g.generation_id
        JOIN requests r ON g.request_id = r.request_id
        '''

        params = []
        if domain:
            query += ' WHERE r.domain = ?'
            params.append(domain)

        cursor.execute(query, params)

        # Initialize counters for each expert in each layer
        activation_counts = {layer: [0] * self.num_experts for layer in range(self.num_layers)}

        # Count activations
        for layer, experts_json in cursor.fetchall():
            activated_experts = json.loads(experts_json)
            for expert in activated_experts:
                activation_counts[layer][expert] += 1

        conn.close()

        return activation_counts

    def get_expert_activation_heatmap(self, domain=None, top_n=20):
        """
        Generate a simple text-based heatmap of expert activation frequency.

        Args:
            domain (str, optional): Domain to filter by
            top_n (int): Number of most frequently activated experts to show per layer

        Returns:
            str: Text-based heatmap representation
        """
        stats = self.get_expert_activation_stats_by_domain(domain)

        heatmap = []
        heatmap.append(f"Expert Activation Heatmap{' for domain: ' + domain if domain else ''}")
        heatmap.append("=" * 80)
        heatmap.append("Layer | Top Activated Experts (expert_id: activation_count)")
        heatmap.append("-" * 80)

        for layer in range(self.num_layers):
            # Get expert activations for this layer
            layer_stats = stats[layer]

            # Sort by activation count
            expert_activations = [(expert_id, count) for expert_id, count in enumerate(layer_stats) if count > 0]
            expert_activations.sort(key=lambda x: x[1], reverse=True)

            # Format the top N experts
            top_experts = expert_activations[:top_n]
            if top_experts:
                expert_str = ", ".join([f"{e_id}:{count}" for e_id, count in top_experts])
            else:
                expert_str = "No activations recorded"

            heatmap.append(f"{layer:5d} | {expert_str}")

        return "\n".join(heatmap)

    def batch_record_generations(self, request_id, tokens, all_expert_activations):
        """
        Record multiple token generations with their expert activations in a single transaction.
        This is more efficient than calling record_generation multiple times.

        Args:
            request_id (int): ID of the related request
            tokens (list): List of generated tokens
            all_expert_activations (list): List of expert activations for each token.
                                          Each item is a list of expert activations per layer.
        """
        if len(tokens) != len(all_expert_activations):
            raise ValueError(
                f"Number of tokens ({len(tokens)}) must match number of expert activation lists ({len(all_expert_activations)})")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            conn.execute("BEGIN TRANSACTION")
            timestamp = datetime.now().isoformat()

            for token_position, (token, expert_activations) in enumerate(zip(tokens, all_expert_activations)):
                if len(expert_activations) != self.num_layers:
                    raise ValueError(
                        f"Expected expert activations for {self.num_layers} layers, got {len(expert_activations)}")

                # Record the token generation
                cursor.execute(
                    'INSERT INTO generations (request_id, token_position, token, timestamp) VALUES (?, ?, ?, ?)',
                    (request_id, token_position, token, timestamp)
                )

                generation_id = cursor.lastrowid

                # Record expert activations for each layer
                for layer, experts in enumerate(expert_activations):
                    # Validate expert indices
                    for expert_idx in experts:
                        if not (0 <= expert_idx < self.num_experts):
                            raise ValueError(f"Expert index {expert_idx} out of range [0, {self.num_experts-1}]")

                    # Store experts as a JSON list
                    experts_json = json.dumps(sorted(experts))

                    cursor.execute(
                        'INSERT INTO expert_activations (generation_id, layer, experts_list) VALUES (?, ?, ?)',
                        (generation_id, layer, experts_json)
                    )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def export_expert_activations_csv(self, output_path, domain=None):
        """
        Export expert activations to a CSV file for further analysis.

        Args:
            output_path (str): Path to save the CSV file
            domain (str, optional): Domain to filter by

        Returns:
            int: Number of rows exported
        """
        import csv

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build the query based on whether domain filtering is needed
        query = '''
        SELECT 
            r.request_id, r.domain, 
            g.generation_id, g.token_position, g.token,
            ea.layer, ea.experts_list
        FROM expert_activations ea
        JOIN generations g ON ea.generation_id = g.generation_id
        JOIN requests r ON g.request_id = r.request_id
        '''

        params = []
        if domain:
            query += ' WHERE r.domain = ?'
            params.append(domain)

        query += ' ORDER BY r.request_id, g.token_position, ea.layer'

        cursor.execute(query, params)

        # Write to CSV
        row_count = 0
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['request_id', 'domain', 'generation_id', 'token_position',
                          'token', 'layer', 'activated_experts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in cursor:
                row_dict = dict(row)
                writer.writerow({
                    'request_id': row_dict['request_id'],
                    'domain': row_dict['domain'],
                    'generation_id': row_dict['generation_id'],
                    'token_position': row_dict['token_position'],
                    'token': row_dict['token'],
                    'layer': row_dict['layer'],
                    'activated_experts': row_dict['experts_list']
                })
                row_count += 1

        conn.close()
        return row_count


class ExpertTrackerRocksDB:
    pass


class ExpertTracker(ExpertTrackerBitmap):
    pass

# Example usage
def example_usage():
    """
    Demonstrates how to use the MoETracker class for tracking expert activations.
    """
    import random

    # Initialize the tracker for a model with 64 layers, 256 experts per layer, 8 active per layer
    tracker = ExpertTracker(db_path='moe_tracking.db')

    # Record a request
    request_text = "Explain the concept of mixture of experts"
    request_id = tracker.record_request(request_text, domain="machine_learning")

    # Simulate generating 5 tokens
    for token_position in range(5):
        # Simulate a generated token
        token = f"token_{token_position}"

        # Simulate expert activations for each layer
        expert_activations = []
        for layer in range(64):
            # Randomly select 8 experts for this layer
            activated_experts = random.sample(range(256), 8)
            expert_activations.append(activated_experts)

        # Record the generation and activations
        tracker.record_generation(
            request_id,
            token_position,
            expert_activations,
            token,
        )

    # Query the data
    print("Example request:")
    requests = tracker.get_requests_by_domain(domain="machine_learning")
    if requests:
        req = requests[0]
        print(f"  Request {req['request_id']}: {req['request_text']} (Domain: {req['domain']})")

        generations = tracker.get_generations(req['request_id'])
        print(f"  Generated {len(generations)} tokens")

        if generations:
            gen = generations[0]
            print(f"  First token: {gen['token']}")

            activations = tracker.get_expert_activations(gen['generation_id'])
            if 0 in activations:
                print(f"  Experts activated in layer 0: {activations[0]}")


if __name__ == "__main__":
    # Run the example
    example_usage()
