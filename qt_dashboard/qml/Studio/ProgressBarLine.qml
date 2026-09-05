import QtQuick
import QtQuick.Layouts
import Studio

ColumnLayout {
    property real value: 0
    property string caption: ""
    property color barColor: Theme.success
    Layout.fillWidth: true
    spacing: 4
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: 10
        radius: 5
        color: Qt.rgba(1, 1, 1, 0.07)
        Rectangle {
            width: Math.max(0, Math.min(1, value)) * parent.width
            height: parent.height
            radius: 5
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0; color: Qt.rgba(barColor.r, barColor.g, barColor.b, 0.65) }
                GradientStop { position: 1; color: barColor }
            }
            Behavior on width { NumberAnimation { duration: 300 } }
        }
    }
    Text {
        text: caption
        visible: caption !== ""
        color: Theme.textMuted
        font.pixelSize: Theme.fsSmall
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }
}
