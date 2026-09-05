import QtQuick
import QtQuick.Layouts
import Studio

// Inline info / warning / error / success note.
Rectangle {
    property string text_: ""
    property string kind: "info"     // info | success | warn | error
    property color tint: kind === "success" ? Theme.green
                       : kind === "warn"    ? Theme.warn
                       : kind === "error"   ? Theme.danger : Theme.accent
    visible: text_ !== ""
    Layout.fillWidth: true
    implicitHeight: body.implicitHeight + 18
    radius: 8
    color: Qt.rgba(tint.r, tint.g, tint.b, 0.10)
    border.color: Qt.rgba(tint.r, tint.g, tint.b, 0.32)
    border.width: 1
    Text {
        id: body
        anchors.fill: parent
        anchors.margins: 9
        text: text_
        color: Theme.textDim
        font.pixelSize: Theme.fsSmall
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
    }
}
