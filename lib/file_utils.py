#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: file utils."""

import os
import sys
import time
import subprocess


def ensure_file_available(file_path, max_wait_time=120):
                   
    if os.path.exists(file_path):
                           
        try:
            if os.path.getsize(file_path) > 0:
                return True
        except:
            pass
    
    print(f"  ファイルがオンラインのみの可能性がある: {os.path.basename(file_path)}")
    print(f"  Dropboxにダウンロードを要求している...")
    
                                                          
    try:
                                           
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir):
                                                 
            try:
                _ = os.listdir(parent_dir)
            except:
                pass
        
                                                         
                                               
        if sys.platform == 'darwin':  # macOS
            try:
                                                        
                subprocess.run(['xattr', '-l', file_path], 
                             capture_output=True, timeout=5, check=False)
            except:
                pass
        
                                  
                                            
        try:
            with open(file_path, 'rb') as f:
                _ = f.read(1)                 
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"  ✓ ファイルが利用可能になりました")
                return True
        except (FileNotFoundError, PermissionError, OSError):
            pass
        
                            
        start_time = time.time()
        check_interval = 2             
        last_progress_time = start_time
        
        while time.time() - start_time < max_wait_time:
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        print(f"  ✓ ファイルがダウンロードされました (サイズ: {file_size:,} bytes)")
                        return True
                except:
                    pass
            
            elapsed = int(time.time() - start_time)
                         
            if time.time() - last_progress_time >= 10:
                print(f"  待機中... ({elapsed}/{max_wait_time}秒) - ファイルにアクセスを試みている...")
                last_progress_time = time.time()
                
                                   
                try:
                    with open(file_path, 'rb') as f:
                        _ = f.read(1)
                except:
                    pass
            
            time.sleep(check_interval)
        
        print(f"  ⚠ タイムアウト: {max_wait_time}秒以内にファイルがダウンロードされなかった")
        print(f"  手動でDropboxアプリからファイルをダウンロードすること: {file_path}")
        return False
        
    except Exception as e:
        print(f"  エラー: {str(e)}")
        return False


def ensure_files_available(file_paths, max_wait_time=60):
    for file_path in file_paths:
        if not ensure_file_available(file_path, max_wait_time):
            return False
    return True


def ensure_directory_exists(directory_path, verbose=True):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            if verbose:
                print(f"Created directory: {directory_path}")
            return True
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    return True
