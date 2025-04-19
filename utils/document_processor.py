class DocumentProcessor:
    """A class for processing and preparing documents for the vector database."""

    @staticmethod
    def prepare_documents(documents, existing_ids, ids, metadatas, papers, papers_added):
        """
        Process and prepare documents for storage in the vector database.

        Args:
            documents (list): List to store document abstracts
            existing_ids (set): Set of existing paper IDs in the database
            ids (list): List to store paper IDs
            metadatas (list): List to store paper metadata
            papers (list): List of papers to process
            papers_added (int): Counter for number of papers added

        Returns:
            int: Number of papers added to the database
        """
        for paper in papers:
            if 'abstract' in paper and paper['abstract'] and paper.get('paperId'):
                paper_id = f"paper_{paper.get('paperId')}"

                # Skip if this paper is already in the database
                if paper_id in existing_ids:
                    continue

                documents.append(paper['abstract'])

                # Create metadata for each paper
                metadata = {
                    'title': paper.get('title', ''),
                    'authors': ', '.join([author.get('name', '') for author in paper.get('authors', [])]),
                    'year': paper.get('year', ''),
                    'url': paper.get('url', ''),
                    'paperId': paper.get('paperId', ''),
                    'local_file_path': paper.get('local_file_path', ''),
                }
                metadatas.append(metadata)
                ids.append(paper_id)
                papers_added += 1
        return papers_added
