    def process_2x2_matrices_optimized(self, matrices: List[np.ndarray], ciphertext: str, 
                                     known_text_path: str = None, chunk_idx: int = 0, 
                                     total_chunks: int = 1) -> Tuple[List[Tuple[np.ndarray, str, float]], bool, int]:
        """
        Process a list of 2x2 matrices with enhanced scoring and early stopping.
        
        Args:
            matrices: List of matrices to process
            ciphertext: Encrypted text
            known_text_path: Path to known plaintext file (optional)
            chunk_idx: Index of current chunk (for logging)
            total_chunks: Total number of chunks (for logging)
            
        Returns:
            Tuple of (results list, found_good_solution flag, matrices_processed count)
        """
        results = []
        found_good_solution = False
        matrices_processed = 0
        
        # Load known text for comparison if available
        known_text = None
        if known_text_path and os.path.exists(known_text_path):
            try:
                with open(known_text_path, 'r', encoding='latin-1') as f:
                    known_text = f.read().upper()
                known_text = re.sub(r'[^A-Z]', '', known_text)
            except Exception as e:
                logging.error(f"Error loading known text: {e}")
        
        # Process each matrix
        start_time = time.time()
        for i, matrix in enumerate(matrices):
            matrices_processed += 1
            
            # Log progress periodically
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start_time
                matrices_per_sec = i / elapsed if elapsed > 0 else 0
                if chunk_idx % 10 == 0:  # Only log for some chunks to avoid log spam
                    logging.debug(f"Chunk {chunk_idx+1}/{total_chunks}: Processed {i}/{len(matrices)} matrices ({matrices_per_sec:.1f} matrices/sec)")
            
            try:
                # Decrypt ciphertext
                decrypted = decrypt_hill(ciphertext, matrix)
                
                # Calculate score using statistical approach
                score = self.score_text(decrypted)
                
                # Check for common Portuguese words
                common_words = ['DE', 'A', 'O', 'QUE', 'E', 'DO', 'DA', 'EM', 'UM', 'PARA', 'COM',
                               'NAO', 'UMA', 'OS', 'NO', 'SE', 'NA', 'POR', 'MAIS', 'AS', 'DOS']
                
                word_count = 0
                found_words = []
                for word in common_words:
                    if word in decrypted:
                        word_count += 1
                        found_words.append(word)
                
                # Bonus for common words
                score += word_count * 0.5
                
                # 4. Compare with known text if available
                if known_text:
                    # Calculate similarity with known text
                    min_len = min(len(decrypted), len(known_text))
                    matches = sum(1 for i in range(min_len) if decrypted[i] == known_text[i])
                    similarity = matches / min_len
                    score += similarity * 10  # Very high weight for similarity
                
                # Add to results if score is positive
                if score > 0:
                    results.append((matrix, decrypted, score))
                    
                    # Check if we found a very good solution (early stopping)
                    # Only stop if we have at least 10 results and a very high score
                    if len(results) >= 10 and score > 15:
                        found_good_solution = True
                        logging.info(f"Found excellent solution in chunk {chunk_idx}: Matrix {matrix}, score: {score:.2f}")
                        logging.info(f"Common words found: {', '.join(found_words)}")
                        logging.info(f"Decrypted text: {decrypted[:50]}...")
                        break
                    
                    # Log high-scoring matrices
                    if score > 10:
                        logging.info(f"Found good solution in chunk {chunk_idx}: Matrix {matrix}, score: {score:.2f}")
                        logging.info(f"Common words found: {', '.join(found_words)}")
                        logging.info(f"Decrypted text: {decrypted[:50]}...")
                    
            except Exception as e:
                if chunk_idx % 10 == 0 and i % 100 == 0:  # Only log occasionally to avoid spam
                    logging.debug(f"Error processing matrix {matrix} in chunk {chunk_idx}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:1000], found_good_solution, matrices_processed  # Return top 1000 candidates from this chunk
