pragma Singleton
import QtQuick

// The dashboard's dark palette — the same tokens the Streamlit CSS uses, so
// both front-ends read as one product.
QtObject {
    readonly property color bg:          "#070b14"
    readonly property color bgAlt:       "#0b1220"
    readonly property color panel:       "#0f172a"
    readonly property color card:        Qt.rgba(1, 1, 1, 0.035)
    readonly property color cardHover:   Qt.rgba(1, 1, 1, 0.06)
    readonly property color border:      Qt.rgba(1, 1, 1, 0.10)
    readonly property color borderStrong:Qt.rgba(56, 189, 248, 0.35)

    readonly property color text:        "#f8fafc"
    readonly property color textDim:     "#cbd5e1"
    readonly property color textMuted:   "#94a3b8"

    readonly property color accent:      "#38bdf8"
    readonly property color success:     "#00e5a3"
    readonly property color green:       "#10b981"
    readonly property color warn:        "#f59e0b"
    readonly property color danger:      "#ef4444"
    readonly property color violet:      "#a78bfa"

    readonly property var series: ["#38bdf8", "#00e5a3", "#f59e0b", "#a78bfa",
                                   "#ef4444", "#22d3ee", "#f472b6", "#84cc16",
                                   "#fb923c", "#60a5fa"]

    readonly property int radius: 12
    readonly property int pad: 14
    readonly property int gap: 12

    readonly property int fsTiny: 11
    readonly property int fsSmall: 12
    readonly property int fsBody: 13
    readonly property int fsTitle: 15
    readonly property int fsBig: 22

    readonly property string mono: "monospace"

    function colorFor(pct) {
        if (pct < 50) return success
        if (pct < 80) return warn
        return danger
    }
}
