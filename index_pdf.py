#!/usr/bin/env python3
"""
Updated script with:
  - FlashAttention disabled for CPU
  - Removed Chroma usage (commented out)
  - Batch PDF-to-image conversion
"""

import json
import os
import pickle
from functools import partial
import gc
import tempfile

##############################################################################
# Disable FlashAttention for CPU usage – set environment variable BEFORE any
# model/library imports that might trigger flash-attn usage.
##############################################################################
os.environ["USE_FLASH_ATTENTION"] = "0"

import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union, Tuple, Any
import argparse

from pdf2image import convert_from_path, pdfinfo_from_path
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn

from pydantic import BaseModel

# Import Byaldi after environment variable is set
from byaldi import RAGMultiModalModel
from byaldi.objects import Result


class PDFMetadata(BaseModel):
    """Metadata for a single PDF document."""
    id: int
    title: str
    file: str
    metadata: Dict = {}  # Optional with empty dict default
    processed: bool
    images_processed: int = 0  # Track number of images processed


class PDFCollection(BaseModel):
    """Collection of PDF documents with metadata."""
    pdfs: List[PDFMetadata]


# Configure rich logging
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration settings."""
    DPI = 300
    CHUNK_SIZE = 10
    MODEL_NAME = "vidore/colqwen2-v1.0"
    DEVICE = "mps"
    IMAGE_FORMAT = "png"
    FILTER_MULTIPLIER = 3  # Get 3x results when filtering to ensure enough after filtering

class PDFIndexer:
    """Indexes PDFs (visually) using Byaldi, storing indexes on disk."""
    
    def __init__(self, pdf_dir: str = "./data/pdfs", index_name: str = "visual_books", index_root: str = ".byaldi"):
        """Initialize the vectorizer with configurable paths.
        
        Args:
            pdf_dir: Directory containing PDFs and metadata
            index_name: Name of the Byaldi index
            index_root: Root directory for storing indexes, defaults to '.byaldi'
        """
        # Resolve paths relative to the current working directory
        pdf_dir_path = Path(pdf_dir).resolve()
        index_root_path = Path(index_root).resolve()
        
        if not pdf_dir_path.exists():
            raise ValueError(f"PDF directory {pdf_dir_path} does not exist")
        if not (pdf_dir_path / "metadata.json").exists():
            raise ValueError(f"metadata.json not found in {pdf_dir_path}")
            
        self.pdf_dir = pdf_dir_path
        self.index_name = index_name
        self.pdf_collection = self._load_pdf_collection()
        self.index_root = index_root_path
        self.index_path = self.index_root / self.index_name

        if self.index_root.exists():
            self.RAG = RAGMultiModalModel.from_index(
                index_path=str(self.index_path),  # Convert Path to string
                device=Config.DEVICE)
        else:
            with tempfile.TemporaryDirectory() as empty_dir:
                self.RAG = RAGMultiModalModel.from_pretrained(
                    pretrained_model_name_or_path=Config.MODEL_NAME,
                    index_root=str(self.index_root),  # Convert Path to string
                    device=Config.DEVICE
                )
                # Create empty index by passing an empty directory
                logger.info("Creating initial empty index...")
                self.RAG.index(
                    input_path=empty_dir,  # Empty directory = empty index
                    index_name=self.index_name,
                    store_collection_with_index=False,
                    overwrite=True
                )
        
        # Pre-fill index_name in the index function
        self.RAG.index = partial(self.RAG.index, index_name=self.index_name)
        self.doc_ids_to_file_names = self.RAG.get_doc_ids_to_file_names()

    def _convert_pdf_to_images(self, pdf: PDFMetadata) -> int:
        image_dir = self._get_image_dir(pdf)
        image_dir.mkdir(exist_ok=True, parents=True)
        
        existing_images = sorted(image_dir.glob("page_*.png"))
        if existing_images:
            logger.info(f"Images already exist for {pdf.file}")
            return len(existing_images)

        logger.info(f"Converting {pdf.file} to images")
        pdf_file_path = self._get_pdf_file_path(pdf)
        
        info = pdfinfo_from_path(str(pdf_file_path))
        max_pages = info["Pages"]
        
        for page in range(1, max_pages + 1, Config.CHUNK_SIZE):
            last_page = min(page + Config.CHUNK_SIZE - 1, max_pages)
            logger.info(f"Processing pages {page} to {last_page}")
            
            images = convert_from_path(
                str(pdf_file_path),
                dpi=Config.DPI,
                fmt=Config.IMAGE_FORMAT,
                first_page=page,
                last_page=last_page
            )
            
            for i, image in enumerate(images, start=page):
                with image:  # Use context manager for each image
                    image.save(image_dir / f"page_{i:04d}.png", "PNG")
            
            del images  # Explicitly clean up the list
        
        return max_pages
    
    def _load_pdf_collection(self) -> PDFCollection:
        """Load and validate metadata from the PDF file.
        
        Returns:
            PDFCollection: Validated collection of PDF metadata
        """
        with open(self.pdf_dir / "metadata.json") as file:
            json_data = json.load(file)  # Use json.load directly on file handle
            return PDFCollection.model_validate(json_data)
    
    def _get_pdf_path(self, pdf: PDFMetadata) -> Path:
        return self.pdf_dir / str(pdf.id)
    
    def _get_pdf_file_path(self, pdf: PDFMetadata) -> Path:
        return self._get_pdf_path(pdf) / pdf.file
    
    def _get_image_dir(self, pdf: PDFMetadata) -> Path:
        return self._get_pdf_path(pdf) / "images"
    
    def _save_pdf_collection(self):
        """Save the PDF collection back to metadata.json"""
        with open(self.pdf_dir / "metadata.json", "w") as f:
            # Convert to dict and save
            json.dump(self.pdf_collection.model_dump(), f, indent=4)

    def _get_page_number(self, image_path: Path) -> int:
        """Extract page number from image filename.
        
        Example: 'page_0001.png' -> 1
        """
        # Extract digits between 'page_' and '.png'
        return int(image_path.stem.split('_')[1])

    def process_pdfs(self) -> None:
        """Process all unprocessed PDFs in the collection with progress tracking."""
        unprocessed_pdfs = [pdf for pdf in self.pdf_collection.pdfs if not pdf.processed]
        logger.info(f"Found {len(unprocessed_pdfs)} unprocessed PDFs")
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            pdf_task = progress.add_task(
                "Processing PDFs...", 
                total=len(unprocessed_pdfs)
            )
            
            for pdf in unprocessed_pdfs:
                pdf_images_path = self._get_image_dir(pdf)
                
                # Convert PDF to images if needed
                self._convert_pdf_to_images(pdf)
                image_paths = sorted(pdf_images_path.glob("page_*.png"))
                
                # Skip already processed images
                remaining_images = image_paths[pdf.images_processed:]

                # Log information about already processed images
                if pdf.images_processed > 0:
                    logger.info(
                        f"PDF {pdf.title} has {pdf.images_processed}/{len(image_paths)} images "
                        f"already processed ({(pdf.images_processed/len(image_paths))*100:.1f}%)"
                    )

                if not remaining_images:
                    logger.info(f"Skipping {pdf.title} - already fully processed")
                    continue
                    
                # Add task for remaining images
                image_task = progress.add_task(
                    f"Processing {pdf.title}",
                    total=len(remaining_images)
                )
                
                for image_path in remaining_images:
                    try:
                        # TODO: I need to pull the chapters from the pdf
                        # and add them as metadata to the index
                        # I should maybe even be doing a LLM call for each page
                        # to get metadata
                        
                        # Create base metadata dict
                        metadata_dict = {
                            "filename": pdf.file,
                            "pdf_title": pdf.title,
                            "page_num": self._get_page_number(image_path)
                        }
                        
                        # Add PDF metadata if it exists and has content
                        if pdf.metadata and isinstance(pdf.metadata, dict):
                            metadata_dict.update(pdf.metadata)
                        
                        # Process single image with combined metadata
                        self.RAG.add_to_index(
                            input_item=image_path,  
                            store_collection_with_index=False,
                            metadata=[metadata_dict]
                        )
                        
                        # Update progress tracking
                        pdf.images_processed += 1
                        self._save_pdf_collection()
                        
                        progress.update(
                            image_task, 
                            advance=1,
                            description=(
                                f"Processing {pdf.title} "
                                f"[{pdf.images_processed}/{len(image_paths)} images • "
                                f"{(pdf.images_processed/len(image_paths))*100:.1f}%]"
                            )
                        )
                        
                        # Force garbage collection after each image
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {str(e)}")
                        # Don't increment images_processed, will retry this image next time
                        break
                
                if pdf.images_processed == len(image_paths):
                    # Only mark as processed if we finished all images
                    pdf.processed = True
                    self._save_pdf_collection()
                elif pdf.images_processed > len(image_paths):
                    # This should never happen - indicates corruption or bug
                    logger.error(
                        f"Critical: PDF {pdf.title} shows {pdf.images_processed} images processed "
                        f"but only {len(image_paths)} images exist. Metadata may be corrupted."
                    )
                    # Reset the counter to match reality
                    pdf.images_processed = 0
                    pdf.processed = False
                    self._save_pdf_collection()
                    raise RuntimeError(
                        f"Image count mismatch for {pdf.title}. Processing aborted for safety."
                    )
                else:
                    # Normal case during error - just log it
                    logger.warning(
                        f"PDF {pdf.title} processing incomplete: "
                        f"{pdf.images_processed}/{len(image_paths)} images processed. "
                        "Will resume in next run."
                    )
                
                # Complete this PDF's task
                progress.update(pdf_task, advance=1)
                progress.remove_task(image_task)
                
                if pdf.processed:
                    logger.info(f"Successfully processed {pdf.title}")
                else:
                    logger.warning(f"Incomplete processing of {pdf.title}, will retry remaining images next run")

    def search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> tuple[List[Result], List[Path]]:
        """Search the indexed PDFs for a given query with optional metadata filtering.
        
        Args:
            query: Search query string
            k: Number of results to return
            metadata_filters: Dict of field:value pairs for exact matches
            metadata_ranges: Dict of field:(min,max) pairs for range matches
            metadata_contains: Dict of field:value pairs for substring matches
        
        Returns:
            Tuple containing list of Result objects and corresponding filenames
        """
        if k < 1:
            raise ValueError("k must be positive")
        if not self.doc_ids_to_file_names:  # Move to __init__
            self.doc_ids_to_file_names = self.RAG.get_doc_ids_to_file_names()
        if not self.doc_ids_to_file_names:
            raise ValueError("No documents indexed")
            
        # Get more results than needed to allow for filtering
        has_filters = any([metadata_filters, metadata_ranges, metadata_contains])
        extra_k = k * Config.FILTER_MULTIPLIER if has_filters else k
        results: Union[List[Result], List[List[Result]]] = self.RAG.search(query, k=extra_k)
        
        # Handle nested results
        if isinstance(results[0], list):
            filtered_results = []
            for result_group in results:
                filtered_group = self._filter_results(
                    result_group,
                    metadata_filters,
                    metadata_ranges,
                    metadata_contains
                )[:k]  # Trim to k after filtering
                if filtered_group:  # Only add groups that have results after filtering
                    filtered_results.append(filtered_group)
            results = filtered_results
        else:
            results = self._filter_results(
                results,
                metadata_filters,
                metadata_ranges,
                metadata_contains
            )[:k]  # Trim to k after filtering
        
        # Match results to filenames
        if isinstance(results[0], list):
            files = [[self.doc_ids_to_file_names[r.doc_id] for r in result_list] 
                    for result_list in results]
        else:
            files = [self.doc_ids_to_file_names[r.doc_id] for r in results]
        
        return results, files
        
    def _filter_results(
        self,
        results: List[Result],
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> List[Result]:
        """Filter results based on metadata criteria.
        
        Args:
            results: List of Result objects to filter
            metadata_filters: Dict of field:value pairs for exact matches
            metadata_ranges: Dict of field:(min,max) pairs for range matches
            metadata_contains: Dict of field:value pairs for substring matches
            
        Returns:
            Filtered list of Result objects
        """
        if not any([metadata_filters, metadata_ranges, metadata_contains]):
            return results
            
        filtered = []
        for result in results:
            # Get metadata from the first (and only) dict in the list
            metadata = self.RAG.model.doc_id_to_metadata[result.doc_id][0]
            
            # Check exact matches
            if metadata_filters and not all(
                metadata.get(field) == value 
                for field, value in metadata_filters.items()
            ):
                continue
                
            # Check ranges
            if metadata_ranges and not all(
                min_val <= metadata.get(field, min_val - 1) <= max_val
                for field, (min_val, max_val) in metadata_ranges.items()
            ):
                continue
                
            # Check contains
            if metadata_contains and not all(
                str(value).lower() in str(metadata.get(field, "")).lower()
                for field, value in metadata_contains.items()
            ):
                continue
                
            filtered.append(result)
            
        return filtered

    def close(self):
        """Properly close and cleanup resources."""
        if hasattr(self, 'RAG'):
            del self.RAG
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _validate_pdf(self, pdf_path: Path) -> bool:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file {pdf_path} not found")
        
        return True

    def update_metadata(self) -> None:
        """Update metadata for all PDFs in the collection to the index."""
        logger.info("Starting metadata update for all PDFs in collection")
        
        # Verify index metadata structure exists
        if not hasattr(self.RAG, 'model') or not hasattr(self.RAG.model, 'doc_id_to_metadata'):
            raise RuntimeError(
                "Index metadata structure not found. This could mean:\n"
                "1. The index was not properly initialized\n"
                "2. The index is corrupted\n"
                "3. No documents have been indexed yet"
            )
        
        # Get processed PDFs
        processed_pdfs = [pdf for pdf in self.pdf_collection.pdfs if pdf.processed]
        logger.info(f"Found {len(processed_pdfs)} processed PDFs out of {len(self.pdf_collection.pdfs)} total")
        
        total_images = 0
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            pdf_task = progress.add_task(
                "Updating PDF metadata...",
                total=len(processed_pdfs)
            )
            
            for pdf in processed_pdfs:
                # Find all images for this PDF
                matching_doc_ids = []
                for doc_id, metadata_list in self.RAG.model.doc_id_to_metadata.items():
                    try:
                        metadata = metadata_list[0]  # Get first (and only) metadata dict
                        if metadata["filename"] == pdf.file:
                            matching_doc_ids.append(doc_id)
                    except Exception as e:
                        logger.error(f"Error processing doc_id {doc_id}: {str(e)}")
                        logger.error(f"Metadata list: {metadata_list}")
                        raise e
                
                if not matching_doc_ids:
                    progress.update(pdf_task, advance=1)
                    raise RuntimeError(f"No images found in index for PDF: {pdf.title}")
                
                if len(matching_doc_ids) != pdf.images_processed:
                    logger.error(
                        f"Image count mismatch for PDF: {pdf.title}\n"
                        f"- Found in index: {len(matching_doc_ids)} images\n"
                        f"- Expected (metadata.json): {pdf.images_processed} images"
                    )
                    progress.update(pdf_task, advance=1)
                    raise RuntimeError(f"Image count mismatch for PDF: {pdf.title}")
                
                logger.info(f"Processing {len(matching_doc_ids)} images for {pdf.title}")
                
                # Add task for images in this PDF
                image_task = progress.add_task(
                    f"Updating {pdf.title}",
                    total=len(matching_doc_ids)
                )
                
                # Update metadata for each matching doc_id
                for doc_id in matching_doc_ids:
                    # Get existing metadata and update with any new fields
                    metadata = self.RAG.model.doc_id_to_metadata[doc_id][0].copy()
                    if pdf.metadata:
                        metadata.update(pdf.metadata)
                    
                    # Store back in ColPali format
                    self.RAG.model.doc_id_to_metadata[doc_id] = [metadata]
                    total_images += 1
                    
                    # Update progress
                    progress.update(
                        image_task,
                        advance=1,
                        description=(
                            f"Updating {pdf.title} metadata"
                        )
                    )
                
                # Complete PDF progress
                progress.update(pdf_task, advance=1)
                progress.remove_task(image_task)
        
        # Save changes to disk
        self.RAG.model._export_index()
        
        # Final summary
        logger.info(
            f"Metadata update complete:\n"
            f"- Processed {len(processed_pdfs)} PDFs\n"
            f"- Updated {total_images} total images\n"
            f"- Average {total_images/len(processed_pdfs):.1f} images per PDF"
        )


def main():
    console = Console()
    
    parser = argparse.ArgumentParser(description="Process PDFs using Byaldi-based visual understanding")
    parser.add_argument("--process", action="store_true", help="Process PDFs for indexing")
    parser.add_argument("--search", help="Search index with query")
    parser.add_argument(
        "--update-metadata", 
        action="store_true",
        help="Update metadata for all processed PDFs in the index"
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        metavar="FIELD=VALUE",
        help="Filter results by exact metadata matches (e.g., --filter page_num=1 filename=book.pdf)"
    )
    parser.add_argument(
        "--range",
        nargs="+",
        metavar="FIELD=MIN,MAX",
        help="Filter results by metadata range (e.g., --range page_num=1,10)"
    )
    parser.add_argument(
        "--contains",
        nargs="+",
        metavar="FIELD=VALUE",
        help="Filter results by metadata substring matches (e.g., --contains title=chapter)"
    )
    parser.add_argument(
        "--index-root",
        default=".byaldi",
        help="Root directory for storing indexes (default: .byaldi)"
    )
    parser.add_argument(
        "--pdf-dir",
        default="./data/pdfs",
        help="Directory containing PDFs and metadata (default: ./data/pdfs)"
    )
    args = parser.parse_args()
    
    try:
        console.print("[cyan]Initializing visual PDF processor...[/cyan]")
        vectorizer = PDFIndexer(pdf_dir=args.pdf_dir, index_root=args.index_root)
        
        if args.process:
            console.print("[cyan]Processing PDFs...[/cyan]")
            vectorizer.process_pdfs()
            console.print(f"[green]Processing Complete![/green]")
        
        if args.update_metadata:
            console.print("[cyan]Updating metadata for all PDFs...[/cyan]")
            vectorizer.update_metadata()
            console.print(f"[green]Metadata Update Complete![/green]")
        
        if args.search:
            # Parse metadata filters
            metadata_filters = {}
            metadata_ranges = {}
            metadata_contains = {}
            
            if args.filter:
                for filter_str in args.filter:
                    try:
                        field, value = filter_str.split('=', 1)
                        # Try to convert to int or float if possible
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                        metadata_filters[field] = value
                    except ValueError:
                        console.print(f"[red]Invalid filter format: {filter_str}[/red]")
                        return
            
            if args.range:
                for range_str in args.range:
                    try:
                        field, range_val = range_str.split('=', 1)
                        min_val, max_val = range_val.split(',', 1)
                        # Try to convert to numbers
                        try:
                            min_val = int(min_val)
                            max_val = int(max_val)
                        except ValueError:
                            try:
                                min_val = float(min_val)
                                max_val = float(max_val)
                            except ValueError:
                                console.print(f"[red]Range values must be numbers: {range_str}[/red]")
                                return
                        metadata_ranges[field] = (min_val, max_val)
                    except ValueError:
                        console.print(f"[red]Invalid range format: {range_str}[/red]")
                        return
            
            if args.contains:
                for contains_str in args.contains:
                    try:
                        field, value = contains_str.split('=', 1)
                        metadata_contains[field] = value
                    except ValueError:
                        console.print(f"[red]Invalid contains format: {contains_str}[/red]")
                        return
            
            # Show active filters
            if any([metadata_filters, metadata_ranges, metadata_contains]):
                console.print("\n[cyan]Active Filters:[/cyan]")
                if metadata_filters:
                    console.print("Exact matches:", metadata_filters)
                if metadata_ranges:
                    console.print("Ranges:", metadata_ranges)
                if metadata_contains:
                    console.print("Contains:", metadata_contains)
            
            console.print(f"\n[cyan]Searching for: {args.search}[/cyan]")
            results, files = vectorizer.search(
                args.search,
                metadata_filters=metadata_filters or None,
                metadata_ranges=metadata_ranges or None,
                metadata_contains=metadata_contains or None
            )
            
            if isinstance(results[0], list):
                # Handle nested results
                for group_idx, (result_group, file_group) in enumerate(zip(results, files), 1):
                    console.print(f"\n[yellow]Result Group {group_idx}:[/yellow]")
                    for result, filename in zip(result_group, file_group):
                        console.print(f"File: [green]{filename}[/green]")
                        metadata = vectorizer.RAG.model.doc_id_to_metadata[result.doc_id][0]
                        console.print(f"Page: {metadata.get('page_num', 'unknown')}")
                        console.print(f"Score: {result.score:.3f}")
                        console.print("Metadata:", metadata)
                        console.print("---")
            else:
                # Handle flat results
                for i, (result, filename) in enumerate(zip(results, files), 1):
                    console.print(f"\n[yellow]Result {i}:[/yellow]")
                    console.print(f"File: [green]{filename}[/green]")
                    metadata = vectorizer.RAG.model.doc_id_to_metadata[result.doc_id][0]
                    console.print(f"Page: {metadata.get('page_num', 'unknown')}")
                    console.print(f"Score: {result.score:.3f}")
                    console.print("Metadata:", metadata)
                    console.print("---")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()