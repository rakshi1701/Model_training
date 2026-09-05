import QtQuick
import QtQuick.Layouts
import Studio

// Multi-series line chart drawn on a Canvas — no QtCharts dependency, so it
// renders the same on every platform/GPU. Data is [{name, points:[{x,y}]}].
Rectangle {
    id: root
    property var seriesData: []
    property string xTitle: "Epoch"
    property string yTitle: ""
    property string emptyText: "No data yet."
    property int decimalsY: 3

    Layout.fillWidth: true
    implicitHeight: 300
    radius: 10
    color: Theme.card
    border.color: Theme.border
    border.width: 1
    clip: true

    readonly property int padL: 58
    readonly property int padR: 14
    readonly property int padT: 24
    readonly property int padB: 30
    property int legendH: legend.implicitHeight + (legend.visible ? 8 : 0)

    property var bounds: computeBounds()
    onSeriesDataChanged: { bounds = computeBounds(); canvas.requestPaint() }
    onWidthChanged: canvas.requestPaint()
    onHeightChanged: canvas.requestPaint()

    function computeBounds() {
        var data = seriesData || []
        var b = { minX: 0, maxX: 1, minY: 0, maxY: 1, empty: true }
        var first = true
        for (var i = 0; i < data.length; ++i) {
            var pts = data[i].points || []
            for (var j = 0; j < pts.length; ++j) {
                var x = pts[j].x, y = pts[j].y
                if (first) { b.minX = b.maxX = x; b.minY = b.maxY = y; first = false }
                else {
                    if (x < b.minX) b.minX = x
                    if (x > b.maxX) b.maxX = x
                    if (y < b.minY) b.minY = y
                    if (y > b.maxY) b.maxY = y
                }
            }
        }
        if (first) return b
        b.empty = false
        if (b.maxX === b.minX) b.maxX = b.minX + 1
        var pad = (b.maxY - b.minY) * 0.08
        if (pad === 0) pad = Math.max(Math.abs(b.maxY) * 0.1, 0.001)
        b.minY -= pad; b.maxY += pad
        return b
    }

    function fmt(v) {
        var a = Math.abs(v)
        if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(1)
        if (a >= 1000) return v.toFixed(0)
        return v.toFixed(root.decimalsY)
    }

    Text {
        anchors.centerIn: parent
        visible: root.bounds.empty
        text: root.emptyText
        color: Theme.textMuted
        font.pixelSize: Theme.fsSmall
    }

    Canvas {
        id: canvas
        anchors.fill: parent
        anchors.bottomMargin: root.legendH
        renderStrategy: Canvas.Immediate
        antialiasing: true
        visible: !root.bounds.empty

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)
            var b = root.bounds
            if (b.empty) return

            var x0 = root.padL, y0 = root.padT
            var w = Math.max(10, width - root.padL - root.padR)
            var h = Math.max(10, height - root.padT - root.padB)

            function px(v) { return x0 + (v - b.minX) / (b.maxX - b.minX) * w }
            function py(v) { return y0 + h - (v - b.minY) / (b.maxY - b.minY) * h }

            // grid + y labels
            ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.06)
            ctx.fillStyle = "#94a3b8"
            ctx.font = "10px sans-serif"
            ctx.lineWidth = 1
            var rows = 5
            for (var i = 0; i <= rows; ++i) {
                var yv = b.minY + (b.maxY - b.minY) * i / rows
                var y = py(yv)
                ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + w, y); ctx.stroke()
                ctx.textAlign = "right"
                ctx.fillText(root.fmt(yv), x0 - 7, y + 3)
            }
            // x labels
            var cols = Math.max(2, Math.min(8, Math.round(b.maxX - b.minX)))
            ctx.textAlign = "center"
            for (var k = 0; k <= cols; ++k) {
                var xv = b.minX + (b.maxX - b.minX) * k / cols
                var x = px(xv)
                ctx.beginPath()
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.04)
                ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); ctx.stroke()
                ctx.fillText(Math.round(xv), x, y0 + h + 15)
            }
            // axis titles
            ctx.fillStyle = "#64748b"
            ctx.textAlign = "right"
            ctx.fillText(root.xTitle, x0 + w, y0 + h + 26)
            if (root.yTitle !== "") {
                ctx.textAlign = "left"
                ctx.fillText(root.yTitle, 6, 12)
            }

            // series
            var data = root.seriesData || []
            for (var s = 0; s < data.length; ++s) {
                var pts = data[s].points || []
                if (!pts.length) continue
                ctx.strokeStyle = Theme.series[s % Theme.series.length]
                ctx.lineWidth = 2
                ctx.lineJoin = "round"
                ctx.beginPath()
                for (var p = 0; p < pts.length; ++p) {
                    var cx = px(pts[p].x), cy = py(pts[p].y)
                    if (p === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
                }
                ctx.stroke()
                if (pts.length === 1) {
                    ctx.fillStyle = Theme.series[s % Theme.series.length]
                    ctx.beginPath()
                    ctx.arc(px(pts[0].x), py(pts[0].y), 3, 0, Math.PI * 2)
                    ctx.fill()
                }
            }

            // hover crosshair
            if (hover.active) {
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.25)
                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.moveTo(hover.px, y0); ctx.lineTo(hover.px, y0 + h)
                ctx.stroke()
            }
        }
    }

    // ---- hover readout ---------------------------------------------------
    QtObject {
        id: hover
        property bool active: false
        property real px: 0
        property real value: 0
    }

    MouseArea {
        anchors.fill: canvas
        hoverEnabled: true
        onExited: { hover.active = false; canvas.requestPaint(); tip.visible = false }
        onPositionChanged: function(mouse) {
            var b = root.bounds
            if (b.empty) return
            var x0 = root.padL
            var w = Math.max(10, canvas.width - root.padL - root.padR)
            if (mouse.x < x0 || mouse.x > x0 + w) { hover.active = false; tip.visible = false; canvas.requestPaint(); return }
            hover.active = true
            hover.px = mouse.x
            hover.value = b.minX + (mouse.x - x0) / w * (b.maxX - b.minX)

            // nearest sample per series
            var lines = [root.xTitle + " " + Math.round(hover.value)]
            var data = root.seriesData || []
            for (var s = 0; s < data.length; ++s) {
                var pts = data[s].points || []
                var best = null, bestD = Number.MAX_VALUE
                for (var p = 0; p < pts.length; ++p) {
                    var d = Math.abs(pts[p].x - hover.value)
                    if (d < bestD) { bestD = d; best = pts[p] }
                }
                if (best) lines.push(data[s].name + ": " + root.fmt(best.y))
            }
            tipText.text = lines.join("\n")
            tip.visible = true
            tip.x = Math.min(mouse.x + 14, root.width - tip.width - 6)
            tip.y = Math.max(6, Math.min(mouse.y + 12, canvas.height - tip.height - 6))
            canvas.requestPaint()
        }
    }

    Rectangle {
        id: tip
        visible: false
        z: 5
        radius: 6
        color: "#0d1526"
        border.color: Theme.border
        border.width: 1
        width: tipText.implicitWidth + 16
        height: tipText.implicitHeight + 12
        Text {
            id: tipText
            anchors.centerIn: parent
            color: Theme.textDim
            font.pixelSize: Theme.fsTiny
            font.family: Theme.mono
        }
    }

    // ---- legend ----------------------------------------------------------
    Flow {
        id: legend
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 8
        spacing: 12
        visible: !root.bounds.empty
        Repeater {
            model: root.seriesData || []
            delegate: Row {
                required property var modelData
                required property int index
                spacing: 5
                Rectangle {
                    width: 10; height: 3; radius: 2
                    anchors.verticalCenter: parent.verticalCenter
                    color: Theme.series[index % Theme.series.length]
                }
                Text {
                    text: modelData.name
                    color: Theme.textMuted
                    font.pixelSize: Theme.fsTiny
                }
            }
        }
    }
}
