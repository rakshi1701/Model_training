import QtQuick
import Studio

Rectangle {
    property string text_: ""
    property color tint: Theme.accent
    implicitWidth: label.implicitWidth + 22
    implicitHeight: 26
    radius: 13
    color: Qt.rgba(tint.r, tint.g, tint.b, 0.13)
    border.color: Qt.rgba(tint.r, tint.g, tint.b, 0.45)
    border.width: 1
    Text {
        id: label
        anchors.centerIn: parent
        text: text_
        color: tint
        font.pixelSize: Theme.fsSmall
        font.bold: true
    }
}
