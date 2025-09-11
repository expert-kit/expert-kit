#!/bin/bash

# Base LFS URL for downloading files
BASE_URL="https://media.githubusercontent.com/media/expert-kit/expert-kit/refs/heads/dev"

echo "Getting LFS file information..."

# Get only unsynced LFS files (where second column is -)
LFS_FILES=$(git lfs ls-files | awk '$2 == "-" {print $3}')

if [ -z "$LFS_FILES" ]; then
    echo "No unsynced LFS files found"
    exit 0
fi

echo "Found unsynced LFS files:"
echo "$LFS_FILES"

echo "Downloading files using curl"

# Download each unsynced LFS file
for file in $LFS_FILES; do
    curl -L "${BASE_URL}/$file?download=true" -o "$file"
    echo "Downloaded $file"
done

echo "Git add downloaded files for tracing"
git add $LFS_FILES

echo "Download and setup completed!"