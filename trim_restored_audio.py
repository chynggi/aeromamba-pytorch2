"""
원본 mixture와 복원된 오디오의 길이를 비교하여 
복원된 오디오가 더 길 경우 불필요한 silence를 제거하는 스크립트
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def load_audio(file_path):
    """오디오 파일을 로드합니다."""
    audio, sr = sf.read(file_path)
    return audio, sr


def save_audio(file_path, audio, sr):
    """오디오 파일을 저장합니다."""
    sf.write(file_path, audio, sr)


def trim_audio_to_match(original_path, restored_path, output_path=None):
    """
    복원된 오디오를 원본 오디오의 길이에 맞춰 자릅니다.
    
    Args:
        original_path: 원본 mixture 파일 경로
        restored_path: 복원된 오디오 파일 경로
        output_path: 출력 파일 경로 (None이면 restored_path를 덮어씁니다)
    
    Returns:
        trimmed: 잘린 오디오가 있으면 True, 없으면 False
    """
    # 오디오 로드
    original_audio, original_sr = load_audio(original_path)
    restored_audio, restored_sr = load_audio(restored_path)
    
    # 샘플링 레이트 확인
    if original_sr != restored_sr:
        print(f"Warning: 샘플링 레이트가 다릅니다. Original: {original_sr}Hz, Restored: {restored_sr}Hz")
    
    # Shape 비교
    original_shape = original_audio.shape
    restored_shape = restored_audio.shape
    
    # 길이 추출 (mono/stereo 모두 고려)
    original_length = original_shape[0] if len(original_shape) == 1 else original_shape[0]
    restored_length = restored_shape[0] if len(restored_shape) == 1 else restored_shape[0]
    
    print(f"Original shape: {original_shape}, length: {original_length}")
    print(f"Restored shape: {restored_shape}, length: {restored_length}")
    
    # 복원된 오디오가 더 긴 경우 자르기
    if restored_length > original_length:
        print(f"복원된 오디오가 {restored_length - original_length} 샘플 더 깁니다. 자르는 중...")
        
        # 원본 길이에 맞춰 자르기
        if len(restored_shape) == 1:
            # Mono
            trimmed_audio = restored_audio[:original_length]
        else:
            # Stereo or multi-channel
            trimmed_audio = restored_audio[:original_length, :]
        
        # 저장
        if output_path is None:
            output_path = restored_path
        
        save_audio(output_path, trimmed_audio, restored_sr)
        print(f"저장 완료: {output_path}")
        return True
    else:
        print("복원된 오디오가 원본보다 짧거나 같습니다. 자르지 않습니다.")
        return False


def process_directory(original_dir, restored_dir, output_dir=None, overwrite=False):
    """
    디렉토리 내의 모든 오디오 파일을 처리합니다.
    
    Args:
        original_dir: 원본 mixture 디렉토리
        restored_dir: 복원된 오디오 디렉토리
        output_dir: 출력 디렉토리 (None이면 restored_dir를 덮어씁니다)
        overwrite: True면 restored_dir를 덮어씁니다, False면 output_dir에 저장합니다
    """
    original_dir = Path(original_dir)
    restored_dir = Path(restored_dir)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 복원된 오디오 파일 목록 가져오기
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    restored_files = []
    for ext in audio_extensions:
        restored_files.extend(restored_dir.glob(f'**/*{ext}'))
    
    print(f"총 {len(restored_files)}개의 파일을 발견했습니다.")
    
    trimmed_count = 0
    skipped_count = 0
    error_count = 0
    
    for restored_file in tqdm(restored_files, desc="처리 중"):
        try:
            # 상대 경로 계산
            rel_path = restored_file.relative_to(restored_dir)
            
            # 원본 파일 경로 찾기
            # 복원된 파일명에서 _restored를 _mixture로 바꿔서 원본 파일명 생성
            # 예: song_sr_000_restored.wav -> song_sr_000_mixture.wav
            restored_stem = restored_file.stem
            
            if "_restored" in restored_stem:
                # _restored를 _mixture로 변경
                original_filename = restored_stem.replace("_restored", "_mixture") + restored_file.suffix
            elif "_mixture" in restored_stem:
                # 이미 _mixture가 있는 경우 그대로 사용
                original_filename = restored_file.name
            else:
                # _restored도 _mixture도 없는 경우 _mixture 추가
                original_filename = restored_stem + "_mixture" + restored_file.suffix
            
            original_file = original_dir / rel_path.parent / original_filename
            
            if not original_file.exists():
                print(f"\n경고: 원본 파일을 찾을 수 없습니다: {original_file}")
                print(f"  복원된 파일: {restored_file}")
                skipped_count += 1
                continue
            
            # 출력 경로 결정
            if overwrite or output_dir is None:
                out_path = restored_file
            else:
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 처리
            trimmed = trim_audio_to_match(
                str(original_file),
                str(restored_file),
                str(out_path)
            )
            
            if trimmed:
                trimmed_count += 1
            
            print()  # 줄바꿈
            
        except Exception as e:
            print(f"\n오류 발생 ({restored_file}): {e}")
            error_count += 1
    
    print("\n" + "="*50)
    print(f"처리 완료!")
    print(f"  - 자른 파일: {trimmed_count}개")
    print(f"  - 건너뛴 파일: {skipped_count}개")
    print(f"  - 오류 발생: {error_count}개")
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="원본 mixture와 복원된 오디오의 길이를 비교하여 불필요한 silence를 제거합니다."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        default=r"O:\THNG\super_resolution_check_mixtures",
        help="원본 mixture 디렉토리 경로"
    )
    parser.add_argument(
        "--restored_dir",
        type=str,
        default=r"O:\Aeromamba_inf",
        help="복원된 오디오 디렉토리 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 경로 (지정하지 않으면 restored_dir를 덮어씁니다)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="복원된 오디오 파일을 덮어쓸지 여부"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("오디오 트리밍 스크립트")
    print("="*50)
    print(f"원본 디렉토리: {args.original_dir}")
    print(f"복원된 디렉토리: {args.restored_dir}")
    
    if args.output_dir:
        print(f"출력 디렉토리: {args.output_dir}")
    else:
        print("출력 디렉토리: 복원된 파일 덮어쓰기")
    
    print("="*50)
    print()
    
    # 디렉토리 존재 확인
    if not os.path.exists(args.original_dir):
        print(f"오류: 원본 디렉토리를 찾을 수 없습니다: {args.original_dir}")
        exit(1)
    
    if not os.path.exists(args.restored_dir):
        print(f"오류: 복원된 디렉토리를 찾을 수 없습니다: {args.restored_dir}")
        exit(1)
    
    # 처리 시작
    process_directory(
        args.original_dir,
        args.restored_dir,
        args.output_dir,
        args.overwrite
    )
