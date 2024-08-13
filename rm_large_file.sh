#!/bin/bash

   # Function to convert size to bytes
   to_bytes() {
       local size=$1
       local unit=${size: -2}
       local number=${size%??}
       case $unit in
           KB) echo $((number * 1024)) ;;
           MB) echo $((number * 1024 * 1024)) ;;
           GB) echo $((number * 1024 * 1024 * 1024)) ;;
           *) echo $number ;;
       esac
   }

   # Set the size threshold (e.g., 50MB)
   threshold=$(to_bytes "50MB")

   # Find large files
   large_files=$(git rev-list --objects --all | 
       git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | 
       sed -n 's/^blob //p' | 
       awk -v threshold=$threshold '$2 >= threshold' | 
       sort -rn -k2 | 
       cut -c 1-12,41-)

   # Remove large files
   for file in $large_files; do
       git filter-branch --force --index-filter \
           "git rm --cached --ignore-unmatch $file" \
           --prune-empty --tag-name-filter cat -- --all
   done

   # Clean up and reclaim space
   rm -rf .git/refs/original/
   git reflog expire --expire=now --all
   git gc --prune=now