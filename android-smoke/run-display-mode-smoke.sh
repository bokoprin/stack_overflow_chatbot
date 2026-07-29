#!/bin/sh
set -eu

RESULTS=android-smoke/results
PACKAGE=jp.openai.upperskeleton.fixed2
ACTIVITY=jp.openai.upperskeleton.MainActivity
mkdir -p "$RESULTS"

# GitHub Actions上の低速エミュレータでSystem UIのANRダイアログが
# アプリ画面へ重ならないよう、OS側のエラーダイアログ表示を抑制する。
adb shell settings put global show_first_crash_dialog 0 || true
adb shell settings put global hide_error_dialogs 1 || true
adb shell settings put global anr_show_background 0 || true
adb shell settings put secure anr_show_background 0 || true
adb logcat -c
adb install -r "$APK"
adb shell am start -W -n "$PACKAGE/$ACTIVITY" > "$RESULTS/start.txt" || true

wait_for_log() {
  pattern="$1"
  limit="$2"
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

assert_no_system_dialog() {
  name="$1"
  adb shell uiautomator dump /sdcard/window.xml >/dev/null 2>&1 || true
  adb pull /sdcard/window.xml "$RESULTS/$name-window.xml" >/dev/null 2>&1 || true
  if [ -s "$RESULTS/$name-window.xml" ] && grep -Eqi "isn.t responding|not responding|応答していません|System UI.*respond" "$RESULTS/$name-window.xml"; then
    echo "Unexpected Android system dialog in $name screenshot"
    cat "$RESULTS/$name-window.xml"
    return 1
  fi
}

wait_for_log 'ANDROID_BONES_ONLY'
grep -q 'ANDROID_BONES_ONLY selection=bone display=bones-only bones=75 muscles=0' "$RESULTS/webview.txt"
sleep 3
assert_no_system_dialog bones-only
adb exec-out screencap -p > "$RESULTS/bones-only.png"

wait_for_log 'ANDROID_MUSCLES_ONLY'
grep -q 'ANDROID_MUSCLES_ONLY selection=muscle display=muscles-only bones=0 muscles=104' "$RESULTS/webview.txt"
sleep 3
assert_no_system_dialog muscles-only
adb exec-out screencap -p > "$RESULTS/muscles-only.png"

wait_for_log 'ANDROID_DISPLAY_DONE'
grep -q 'ANDROID_DISPLAY_DONE selection=bone display=related related=true' "$RESULTS/webview.txt"
sleep 3
assert_no_system_dialog related
adb exec-out screencap -p > "$RESULTS/related.png"

adb shell pidof "$PACKAGE" > "$RESULTS/pid.txt"
test -s "$RESULTS/pid.txt"
adb logcat -d > "$RESULTS/logcat.txt"
! grep -q 'ANDROID_ANATOMY_ERROR' "$RESULTS/webview.txt"
! grep -q 'FATAL EXCEPTION' "$RESULTS/logcat.txt"
adb shell dumpsys meminfo "$PACKAGE" > "$RESULTS/meminfo.txt"
