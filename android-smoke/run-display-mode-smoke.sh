#!/bin/sh
set -eu

RESULTS=android-smoke/results
PACKAGE=jp.openai.upperskeleton.fixed2
ACTIVITY=jp.openai.upperskeleton.MainActivity
COMPONENT="$PACKAGE/$ACTIVITY"
mkdir -p "$RESULTS"

# 高精細モデルの読み込み待ち中にエミュレータ画面が消灯しないようにする。
adb shell svc power stayon true || true
adb shell settings put system screen_off_timeout 2147483647 || true
adb shell input keyevent 224 || true
adb shell wm dismiss-keyguard || true
adb shell input keyevent 82 || true

# 低速エミュレータでSystem UIのANRダイアログが重ならないよう抑制する。
adb shell settings put global show_first_crash_dialog 0 || true
adb shell settings put global hide_error_dialogs 1 || true
adb shell settings put global anr_show_background 0 || true
adb shell settings put secure anr_show_background 0 || true
adb logcat -c
adb install -r "$APK"
adb shell am start -W -n "$COMPONENT" > "$RESULTS/start.txt" || true

wait_for_log() {
  pattern="$1"
  limit="${2:-120}"
  i=0
  while [ "$i" -lt "$limit" ]; do
    adb logcat -d -s SkeletonWebView:I '*:S' > "$RESULTS/webview.txt" || true
    if grep -q "$pattern" "$RESULTS/webview.txt"; then
      return 0
    fi
    if grep -q 'ANDROID_ANATOMY_ERROR' "$RESULTS/webview.txt"; then
      cat "$RESULTS/webview.txt"
      return 1
    fi
    i=$((i + 1))
    sleep 2
  done
  echo "Timed out waiting for: $pattern"
  cat "$RESULTS/webview.txt" || true
  return 1
}

wake_screen() {
  # Activityは再起動せず、表示モードを保ったまま画面だけを点灯する。
  adb shell input keyevent 224 || true
  adb shell wm dismiss-keyguard || true
  adb shell input keyevent 82 || true
  sleep 2
  adb shell dumpsys power > "$RESULTS/power.txt" || true
  adb shell dumpsys window > "$RESULTS/window-state.txt" || true
}

prepare_clean_screen() {
  name="$1"
  wake_screen
  attempt=0
  while [ "$attempt" -lt 3 ]; do
    adb shell uiautomator dump /sdcard/window.xml >/dev/null 2>&1 || true
    adb pull /sdcard/window.xml "$RESULTS/$name-window.xml" >/dev/null 2>&1 || true
    if [ ! -s "$RESULTS/$name-window.xml" ] || ! grep -Eqi "isn.t responding|not responding|応答していません|System UI.*respond" "$RESULTS/$name-window.xml"; then
      return 0
    fi
    echo "Dismissing System UI ANR dialog before $name screenshot"
    adb shell input tap 540 1058 || true
    sleep 6
    wake_screen
    attempt=$((attempt + 1))
  done
  adb shell uiautomator dump /sdcard/window.xml >/dev/null 2>&1 || true
  adb pull /sdcard/window.xml "$RESULTS/$name-window.xml" >/dev/null 2>&1 || true
  if [ -s "$RESULTS/$name-window.xml" ] && grep -Eqi "isn.t responding|not responding|応答していません|System UI.*respond" "$RESULTS/$name-window.xml"; then
    echo "Android system dialog remained in $name screenshot"
    cat "$RESULTS/$name-window.xml"
    return 1
  fi
}

wait_for_log 'ANDROID_FRAME_STATS mode=bones-only'
grep -E 'ANDROID_FRAME_STATS mode=bones-only .*pass=true' "$RESULTS/webview.txt"
prepare_clean_screen bones-only
adb exec-out screencap -p > "$RESULTS/bones-only.png"

wait_for_log 'ANDROID_FRAME_STATS mode=muscles-only'
grep -E 'ANDROID_FRAME_STATS mode=muscles-only .*pass=true' "$RESULTS/webview.txt"
prepare_clean_screen muscles-only
adb exec-out screencap -p > "$RESULTS/muscles-only.png"

wait_for_log 'ANDROID_FRAME_STATS mode=related'
grep -E 'ANDROID_FRAME_STATS mode=related .*pass=true' "$RESULTS/webview.txt"
prepare_clean_screen related
adb exec-out screencap -p > "$RESULTS/related.png"

adb shell pidof "$PACKAGE" > "$RESULTS/pid.txt"
test -s "$RESULTS/pid.txt"
adb logcat -d > "$RESULTS/logcat.txt"
! grep -q 'ANDROID_ANATOMY_ERROR' "$RESULTS/webview.txt"
! grep -q 'FATAL EXCEPTION' "$RESULTS/logcat.txt"
adb shell dumpsys meminfo "$PACKAGE" > "$RESULTS/meminfo.txt"
