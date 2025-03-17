# Sail Chat Tools

A Python-based tool for indexing and searching PDF documents using visual understanding. This tool uses Byaldi's multimodal model to create a searchable index of PDF documents, allowing for semantic search across both text and visual content.

## Features

- PDF to image conversion with configurable DPI
- Visual understanding of PDF content using Byaldi's multimodal model
- Semantic search across PDF documents
- Metadata filtering and range-based search
- Progress tracking and rich console output
- Batch processing of PDFs with resume capability

## Prerequisites

- Python 3.10 or higher
- Poppler (required for PDF processing)
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler-windows](http://blog.alivate.com.au/poppler-windows/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/celtiberi/sail-chat-tools.git
cd sail-chat-tools
```

2. Create and activate a conda environment:
```bash
conda create -n sail-chat-tools python=3.11
conda activate sail-chat-tools
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the data directory structure:
```bash
mkdir -p data/pdfs
```

5. Create a `metadata.json` file in the `data/pdfs` directory with the following structure:
```json
{
    "pdfs": [
        {
            "id": 1,
            "title": "Your PDF Title",
            "file": "your-pdf-file.pdf",
            "metadata": {},
            "processed": false,
            "images_processed": 0
        }
    ]
}
```

## Usage

### Processing PDFs

To process and index PDFs:
```bash
python index_pdf.py --process
```

To process PDFs from a custom directory:
```bash
python index_pdf.py --process --pdf-dir /path/to/your/pdfs
```

### Searching

Basic search:
```bash
python index_pdf.py --search "your search query"
```

Search in a custom PDF directory:
```bash
python index_pdf.py --search "query" --pdf-dir /path/to/your/pdfs
```

Search with metadata filters:
```bash
# Filter by exact matches
python index_pdf.py --search "query" --filter page_num=1 filename=book.pdf

# Filter by range
python index_pdf.py --search "query" --range page_num=1,10

# Filter by substring match
python index_pdf.py --search "query" --contains title=chapter
```

### Updating Metadata

To update metadata for all processed PDFs:
```bash
python index_pdf.py --update-metadata
```

To update metadata from a custom directory:
```bash
python index_pdf.py --update-metadata --pdf-dir /path/to/your/pdfs
```

### Custom Index Location

To specify a custom location for the index:
```bash
python index_pdf.py --process --index-root /path/to/custom/index/dir
```

To specify both custom PDF directory and index location:
```bash
python index_pdf.py --process --pdf-dir /path/to/your/pdfs --index-root /path/to/custom/index/dir
```

## Configuration

Key configuration settings in `Config` class:
- `DPI`: Resolution for PDF to image conversion (default: 300)
- `CHUNK_SIZE`: Number of pages to process in each batch (default: 10)
- `MODEL_NAME`: Byaldi model to use (default: "vidore/colqwen2-v1.0")
- `DEVICE`: Device to run the model on (default: "mps")
- `IMAGE_FORMAT`: Format for converted images (default: "png")

## Directory Structure

```
sail-chat-tools/
├── data/
│   └── pdfs/
│       ├── metadata.json
│       └── [pdf_id]/
│           ├── [pdf_file]
│           └── images/
│               └── page_*.png
├── .byaldi/              # Created automatically
├── index_pdf.py
├── requirements.txt
└── README.md
```

## Notes

- The script creates an 'images' subdirectory for each PDF to store converted images
- Processing can be resumed if interrupted - the script tracks progress in metadata.json
- The .byaldi directory contains the search index and will be created automatically
- Large PDFs may take significant time to process due to high DPI conversion

## License

[Add your license information here] 