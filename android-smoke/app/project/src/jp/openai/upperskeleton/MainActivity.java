package jp.openai.upperskeleton;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.util.Log;
import android.webkit.ConsoleMessage;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final String APP_ORIGIN = "https://appassets.androidplatform.net/";
    private static final String START_PAGE = APP_ORIGIN + "index.html";
    private FrameLayout root;
    private WebView webView;
    private boolean safeMode;
    private boolean rendererRecoveryAttempted;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            root = new FrameLayout(this);
            root.setBackgroundColor(Color.rgb(7, 16, 29));
            setContentView(root);
            attachFreshWebView(savedInstanceState);
        } catch (Throwable error) {
            showNativeFallback("アプリの初期化に失敗しました", error);
        }
    }

    private void attachFreshWebView(Bundle savedInstanceState) {
        WebView next = createWebView();
        webView = next;
        root.removeAllViews();
        root.addView(next, new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ));
        boolean restored = savedInstanceState != null && next.restoreState(savedInstanceState) != null;
        if (!restored) loadAppPage(next);
    }

    @SuppressLint("SetJavaScriptEnabled")
    private WebView createWebView() {
        WebView view = new WebView(this);
        view.setBackgroundColor(Color.rgb(7, 16, 29));
        view.setWebChromeClient(new LoggingWebChromeClient());
        view.setWebViewClient(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
            ? new Api26SkeletonWebViewClient(this)
            : new BaseSkeletonWebViewClient(this));

        WebSettings settings = view.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setSupportZoom(false);
        settings.setUserAgentString(settings.getUserAgentString() + " FullSkeletonAtlas/1.2.0");
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Api26.configureWebView(view, settings);
        }
        return view;
    }

    private void loadAppPage(WebView target) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getAssets().open("index.html"), StandardCharsets.UTF_8))) {
            StringBuilder html = new StringBuilder(100_000);
            String line;
            while ((line = reader.readLine()) != null) {
                html.append(line).append('\n');
            }
            boolean smokeTest = getIntent() != null && getIntent().getBooleanExtra("smoke_test", false);
            String bootstrap = "<script>" +
                "window.__ANDROID_APP__=true;" +
                "window.__ANDROID_SAFE_MODE__=" + safeMode + ";" +
                "window.__ANDROID_SMOKE_TEST__=" + smokeTest + ";" +
                "</script>";
            int head = html.indexOf("<head>");
            if (head >= 0) html.insert(head + "<head>".length(), bootstrap);
            target.loadDataWithBaseURL(START_PAGE, html.toString(), "text/html", "UTF-8", null);
        } catch (IOException error) {
            showNativeFallback("アプリデータを読み込めませんでした", error);
        }
    }

    private WebResourceResponse openBundledAsset(WebResourceRequest request) {
        if (request == null || request.getUrl() == null) return null;
        String url = request.getUrl().toString();
        if (!url.startsWith(APP_ORIGIN)) return null;
        String path = request.getUrl().getPath();
        if (path == null || path.length() <= 1) return null;
        path = path.substring(1);
        if (path.contains("..") || path.startsWith("/")) return null;
        try {
            InputStream data = getAssets().open(path);
            return new WebResourceResponse(mimeTypeFor(path), encodingFor(path), data);
        } catch (IOException ignored) {
            return null;
        }
    }

    private static String mimeTypeFor(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".js") || lower.endsWith(".mjs")) return "text/javascript";
        if (lower.endsWith(".json")) return "application/json";
        if (lower.endsWith(".html")) return "text/html";
        if (lower.endsWith(".css")) return "text/css";
        if (lower.endsWith(".stl")) return "model/stl";
        if (lower.endsWith(".glb")) return "model/gltf-binary";
        return "application/octet-stream";
    }

    private static String encodingFor(String path) {
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".js") || lower.endsWith(".mjs") || lower.endsWith(".json") ||
            lower.endsWith(".html") || lower.endsWith(".css")) return "UTF-8";
        return null;
    }

    private void handleRenderProcessGone(WebView failedView, boolean didCrash) {
        if (failedView != null) {
            ViewGroup parent = (ViewGroup) failedView.getParent();
            if (parent != null) parent.removeView(failedView);
            failedView.setWebChromeClient(null);
            failedView.setWebViewClient(null);
            failedView.destroy();
        }
        if (webView == failedView) webView = null;
        safeMode = true;
        if (!rendererRecoveryAttempted) {
            rendererRecoveryAttempted = true;
            Toast.makeText(this,
                didCrash ? "3D描画を安定モードで再起動します" : "3D描画を軽量モードで再起動します",
                Toast.LENGTH_LONG).show();
            try {
                attachFreshWebView(null);
                return;
            } catch (Throwable error) {
                showNativeFallback("3D描画の再起動に失敗しました", error);
                return;
            }
        }
        showNativeFallback("3D描画を開始できませんでした", null);
    }

    private void showNativeFallback(String title, Throwable error) {
        if (root == null) {
            root = new FrameLayout(this);
            root.setBackgroundColor(Color.rgb(7, 16, 29));
            setContentView(root);
        }
        if (webView != null) {
            try {
                webView.stopLoading();
                webView.destroy();
            } catch (Throwable ignored) {
                // 既に破棄されたWebViewでは何もしない。
            }
            webView = null;
        }
        TextView message = new TextView(this);
        message.setTextColor(Color.WHITE);
        message.setTextSize(16);
        message.setGravity(Gravity.CENTER);
        message.setPadding(48, 48, 48, 48);
        String detail = error == null ? "" : "\n\n" + error.getClass().getSimpleName() + ": " + String.valueOf(error.getMessage());
        message.setText(title + "\n\nアプリを再起動してください。問題が続く場合は、この画面を添えて報告してください。" + detail);
        root.removeAllViews();
        root.addView(message, new FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ));
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        if (webView != null) webView.saveState(outState);
        super.onSaveInstanceState(outState);
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (webView != null) {
            try {
                webView.loadUrl("about:blank");
                webView.stopLoading();
                webView.setWebChromeClient(null);
                webView.setWebViewClient(null);
                webView.destroy();
            } catch (Throwable ignored) {
                // 終了処理ではクラッシュを再送出しない。
            }
            webView = null;
        }
        super.onDestroy();
    }

    private static final class LoggingWebChromeClient extends WebChromeClient {
        @Override
        public boolean onConsoleMessage(ConsoleMessage consoleMessage) {
            if (consoleMessage != null) {
                Log.i("SkeletonWebView", consoleMessage.message());
            }
            return true;
        }
    }

    private static class BaseSkeletonWebViewClient extends WebViewClient {
        final MainActivity activity;

        BaseSkeletonWebViewClient(MainActivity activity) {
            this.activity = activity;
        }

        @Override
        public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
            WebResourceResponse response = activity.openBundledAsset(request);
            return response != null ? response : super.shouldInterceptRequest(view, request);
        }

        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            if (!request.isForMainFrame()) return false;
            return !request.getUrl().toString().startsWith(APP_ORIGIN);
        }
    }

    private static final class Api26SkeletonWebViewClient extends BaseSkeletonWebViewClient {
        Api26SkeletonWebViewClient(MainActivity activity) {
            super(activity);
        }

        @Override
        public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
            activity.handleRenderProcessGone(view, detail != null && detail.didCrash());
            return true;
        }
    }

    private static final class Api26 {
        static void configureWebView(WebView view, WebSettings settings) {
            try {
                settings.setSafeBrowsingEnabled(true);
                view.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_BOUND, false);
            } catch (RuntimeException ignored) {
                // 古いWebView実装でも起動を継続する。
            }
        }
    }
}
