#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


def make_filter(sr):
    def filt(data, cutoff, btype="low", order=3):
        nyq = sr / 2

        if isinstance(cutoff, (list, tuple)):
            wn = [c / nyq for c in cutoff]
        else:
            wn = cutoff / nyq

        b, a = butter(order, wn, btype=btype)
        return filtfilt(b, a, data)

    return filt


def normalize(data):
    peak = np.max(np.abs(data))
    return data / (peak + 1e-12)


def generate_weather(duration=60.0, sr=44100, output=None):
    # シードを固定しないため、実行するたびに結果が変わる
    rng = np.random.default_rng()

    sample_count = int(sr * duration)
    time = np.arange(sample_count) / sr
    filt = make_filter(sr)

    left = np.zeros(sample_count, dtype=np.float64)
    right = np.zeros(sample_count, dtype=np.float64)

    # 小雨の性質を毎回ランダムに決める
    rain_brightness = rng.uniform(0.85, 1.15)
    rain_density = rng.uniform(0.65, 1.10)
    slow_period = rng.uniform(10.0, 18.0)

    for channel, phase in [
        (left, rng.uniform(0, 2 * np.pi)),
        (right, rng.uniform(0, 2 * np.pi)),
    ]:
        white_noise = rng.standard_normal(sample_count)

        # 細い雨のサーッという音
        fine_rain = filt(
            white_noise,
            [
                int(800 * rain_brightness),
                int(9000 * rain_brightness),
            ],
            btype="band",
            order=2,
        )
        fine_rain = normalize(fine_rain)

        # 雨の低い胴鳴り
        rain_body = filt(
            rng.standard_normal(sample_count),
            [160, 1300],
            btype="band",
            order=2,
        )
        rain_body = normalize(rain_body)

        # 雨量のゆっくりした変化
        slow_noise = filt(
            rng.standard_normal(sample_count),
            0.35,
            btype="low",
            order=2,
        )
        slow_noise = normalize(slow_noise)

        intensity = (
            rng.uniform(0.38, 0.50)
            + rng.uniform(0.07, 0.13) * slow_noise
            + rng.uniform(0.03, 0.07)
            * np.sin(
                2 * np.pi * time / slow_period + phase
            )
        )

        intensity = np.clip(intensity, 0.22, 0.68)

        channel += (
            rng.uniform(0.038, 0.055)
            * fine_rain
            * intensity
        )

        channel += (
            rng.uniform(0.008, 0.015)
            * rain_body
            * (0.75 + 0.25 * intensity)
        )

    # 雨粒の数、位置、強さ、左右配置も毎回変える
    drop_count = int(
        duration
        * rng.uniform(8.0, 16.0)
        * rain_density
    )

    drop_times = rng.uniform(
        0.0,
        duration,
        drop_count,
    )

    for drop_time in drop_times:
        start = int(drop_time * sr)

        event_length = rng.uniform(0.003, 0.020)
        event_samples = max(
            8,
            int(event_length * sr),
        )

        local_time = np.arange(event_samples) / sr

        noise = rng.standard_normal(event_samples)

        drop = filt(
            noise,
            [1800, 12000],
            btype="band",
            order=2,
        )
        drop = normalize(drop)

        envelope = np.exp(
            -local_time / rng.uniform(0.002, 0.009)
        )

        amplitude = rng.uniform(0.003, 0.026)

        # たまに少し強い雨粒
        if rng.random() < 0.025:
            amplitude *= rng.uniform(1.6, 2.8)

        event = amplitude * drop * envelope

        # 左右の位置
        pan = rng.uniform(0.0, 1.0)

        left_gain = np.cos(
            pan * np.pi / 2
        )

        right_gain = np.sin(
            pan * np.pi / 2
        )

        end = min(
            start + event_samples,
            sample_count,
        )

        event = event[:end - start]

        left[start:end] += event * left_gain
        right[start:end] += event * right_gain

    def add_thunder(
        center_time,
        strength,
        pan,
        closer,
    ):
        total_length = rng.uniform(8.0, 15.0)
        thunder_samples = int(total_length * sr)

        start = int(
            max(
                0,
                center_time - rng.uniform(0.8, 1.5),
            )
            * sr
        )

        if start >= sample_count:
            return

        local_time = (
            np.arange(thunder_samples) / sr
        )

        # 雷の低音
        low_noise = filt(
            rng.standard_normal(thunder_samples),
            rng.uniform(100, 140),
            btype="low",
            order=4,
        )
        low_noise = normalize(low_noise)

        # ゴロゴロした中域
        mid_noise = filt(
            rng.standard_normal(thunder_samples),
            [90, rng.uniform(900, 1400)],
            btype="band",
            order=2,
        )
        mid_noise = normalize(mid_noise)

        # 最初の雷鳴の輪郭
        crack_noise = filt(
            rng.standard_normal(thunder_samples),
            [600, rng.uniform(2800, 4200)],
            btype="band",
            order=2,
        )
        crack_noise = normalize(crack_noise)

        # 雷鳴が何度か転がるような山
        centers = [
            rng.uniform(0.7, 1.2),
            rng.uniform(2.3, 3.6),
            rng.uniform(4.8, 6.8),
            rng.uniform(7.5, 10.0),
        ]

        widths = [
            rng.uniform(0.45, 0.85),
            rng.uniform(1.0, 1.8),
            rng.uniform(1.7, 2.6),
            rng.uniform(2.0, 3.2),
        ]

        weights = [
            rng.uniform(0.95, 1.25),
            rng.uniform(0.65, 0.95),
            rng.uniform(0.42, 0.70),
            rng.uniform(0.20, 0.42),
        ]

        envelope = np.zeros(thunder_samples)

        for center, width, weight in zip(
            centers,
            widths,
            weights,
        ):
            envelope += (
                weight
                * np.exp(
                    -(
                        (
                            local_time - center
                        )
                        / width
                    )
                    ** 2
                )
            )

        envelope *= np.exp(
            -local_time
            / rng.uniform(13.0, 18.0)
        )

        # 雷の超低音成分
        frequencies = rng.uniform(
            [27, 39, 52],
            [35, 49, 66],
        )

        phases = rng.uniform(
            0,
            2 * np.pi,
            3,
        )

        sub = (
            np.sin(
                2
                * np.pi
                * frequencies[0]
                * local_time
                + phases[0]
            )
            + rng.uniform(0.45, 0.75)
            * np.sin(
                2
                * np.pi
                * frequencies[1]
                * local_time
                + phases[1]
            )
            + rng.uniform(0.20, 0.38)
            * np.sin(
                2
                * np.pi
                * frequencies[2]
                * local_time
                + phases[2]
            )
        )

        transient_decay = rng.uniform(
            0.15,
            0.25 if closer else 0.45,
        )

        transient_envelope = np.exp(
            -local_time / transient_decay
        )

        transient = (
            crack_noise
            * transient_envelope
        )

        thunder = strength * (
            rng.uniform(0.18, 0.25)
            * low_noise
            * envelope
            + rng.uniform(0.040, 0.065)
            * mid_noise
            * envelope
            + rng.uniform(0.026, 0.042)
            * sub
            * envelope
            + rng.uniform(0.020, 0.045)
            * transient
        )

        left_gain = np.cos(
            pan * np.pi / 2
        )

        right_gain = np.sin(
            pan * np.pi / 2
        )

        end = min(
            start + thunder_samples,
            sample_count,
        )

        segment = thunder[:end - start]

        left[start:end] += (
            segment * left_gain
        )

        right[start:end] += (
            segment * right_gain
        )

        # 雷鳴の反射も毎回ランダム
        reflection_count = int(
            rng.integers(2, 5)
        )

        for _ in range(reflection_count):
            delay_seconds = rng.uniform(
                0.12,
                0.95,
            )

            gain = rng.uniform(
                0.05,
                0.18,
            )

            delayed_start = (
                start
                + int(delay_seconds * sr)
            )

            if delayed_start >= sample_count:
                continue

            delayed_end = min(
                delayed_start + len(segment),
                sample_count,
            )

            delayed = (
                segment[
                    :delayed_end
                    - delayed_start
                ]
                * gain
            )

            if pan < 0.5:
                right[
                    delayed_start:delayed_end
                ] += delayed
            else:
                left[
                    delayed_start:delayed_end
                ] += delayed

    # 雷は毎回2回から5回
    thunder_count = int(
        rng.integers(2, 6)
    )

    safe_start = min(
        5.0,
        duration * 0.15,
    )

    safe_end = max(
        safe_start + 1.0,
        duration - 5.0,
    )

    thunder_times = np.sort(
        rng.uniform(
            safe_start,
            safe_end,
            thunder_count,
        )
    )

    for thunder_time in thunder_times:
        add_thunder(
            center_time=float(thunder_time),
            strength=float(
                rng.uniform(0.85, 1.45)
            ),
            pan=float(
                rng.uniform(0.08, 0.92)
            ),
            closer=bool(
                rng.random() < 0.35
            ),
        )

    # 薄い風
    wind = filt(
        rng.standard_normal(sample_count),
        [40, 450],
        btype="band",
        order=2,
    )
    wind = normalize(wind)

    wind_period = rng.uniform(
        12.0,
        22.0,
    )

    wind_envelope = (
        rng.uniform(0.32, 0.46)
        + rng.uniform(0.14, 0.24)
        * np.sin(
            2
            * np.pi
            * time
            / wind_period
            + rng.uniform(0, 2 * np.pi)
        )
    )

    left += (
        rng.uniform(0.0035, 0.0065)
        * wind
        * wind_envelope
    )

    right += (
        rng.uniform(0.0035, 0.0065)
        * np.roll(
            wind,
            int(
                rng.uniform(0.010, 0.030)
                * sr
            ),
        )
        * wind_envelope
    )

    # ステレオ化
    stereo = np.column_stack(
        (left, right)
    )

    # 軽い音圧調整
    stereo = np.tanh(
        stereo
        * rng.uniform(1.45, 1.70)
    )

    stereo /= (
        np.max(np.abs(stereo))
        + 1e-12
    )

    stereo *= 0.94

    # 出力名を省略すると日時入りになる
    if output is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        output = Path(
            f"random_light_rain_thunder_"
            f"{timestamp}.wav"
        )
    else:
        output = Path(output)

    wavfile.write(
        output,
        sr,
        (stereo * 32767).astype(np.int16),
    )

    return (
        output,
        thunder_count,
        thunder_times,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "実行するたびに違う"
            "小雨と雷のWAVを生成します。"
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="音声の長さ。初期値は60秒。",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "出力ファイル名。"
            "省略すると日時入りになります。"
        ),
    )

    args = parser.parse_args()

    output, thunder_count, thunder_times = (
        generate_weather(
            duration=args.duration,
            output=args.output,
        )
    )

    print(f"Created: {output}")
    print(
        f"Thunder count: "
        f"{thunder_count}"
    )

    print(
        "Thunder times:",
        ", ".join(
            f"{time:.1f}s"
            for time in thunder_times
        ),
    )


if __name__ == "__main__":
    main()
