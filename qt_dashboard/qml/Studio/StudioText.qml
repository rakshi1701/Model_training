import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

TextField {
    id: ctl
    Layout.fillWidth: true
    implicitHeight: 34
    color: Theme.text
    font.pixelSize: Theme.fsBody
    placeholderTextColor: Qt.rgba(0.58, 0.65, 0.72, 0.7)
    selectByMouse: true
    leftPadding: 10
    background: Rectangle {
        radius: 8
        color: Theme.panel
        border.color: ctl.activeFocus ? Theme.borderStrong : Theme.border
        border.width: 1
    }
}
