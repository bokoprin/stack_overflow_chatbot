#!/bin/sh
set -eu

mkdir -p android-smoke/results
adb logcat -c
adb install -r "$APK"
adb shell am start -W -n jp.openai.upperskeleton.fixed2/jp.openai.upperskeleton.MainActivity --ez smoke_test true \
  | tee android-smoke/results/am-start.txt

i=1
while [ "$i" -le 90 ]; do
  adb logcat -d -s SkeletonWebView:I '*:S' > android-smoke/results/webview-log.txt || true
  if grep -q 'SKELETON_SMOKE_OK selected=femur-左' android-smoke/results/webview-log.txt; then
    break
  fi
  sleep 2
  i=$((i + 1))
done

adb shell pidof jp.openai.upperskeleton.fixed2 | tee android-smoke/results/pid.txt
test -s android-smoke/results/pid.txt
adb logcat -d > android-smoke/results/logcat.txt
grep -q 'SKELETON_RUNTIME_READY bones=75 selectable=75' android-smoke/results/webview-log.txt
grep -q 'SKELETON_SMOKE_OK selected=femur-左 mark=記録から外す note=Androidエミュレーター自動確認' android-smoke/results/webview-log.txt
! grep -q 'FATAL EXCEPTION' android-smoke/results/logcat.txt
! grep -q 'AndroidRuntime.*FATAL' android-smoke/results/logcat.txt
adb exec-out screencap -p > android-smoke/results/emulator.png
adb shell uiautomator dump /sdcard/window.xml || true
adb pull /sdcard/window.xml android-smoke/results/window.xml || true
adb shell dumpsys meminfo jp.openai.upperskeleton.fixed2 > android-smoke/results/meminfo.txt
