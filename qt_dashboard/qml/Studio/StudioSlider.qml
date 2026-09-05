import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

Slider {
    id: ctl
    Layout.fillWidth: true
    implicitHeight: 26
    background: Rectangle {
        x: 0
        y: (ctl.height - height) / 2
        width: ctl.availableWidth
        height: 5
        radius: 3
        color: Qt.rgba(1, 1, 1, 0.08)
        Rectangle {
            width: ctl.visualPosition * parent.width
            height: parent.height
            radius: 3
            color: Theme.accent
        }
    }
    handle: Rectangle {
        x: ctl.leftPadding + ctl.visualPosition * (ctl.availableWidth - width)
        y: (ctl.height - height) / 2
        width: 16; height: 16; radius: 8
        color: ctl.pressed ? Qt.lighter(Theme.accent, 1.2) : Theme.accent
        border.color: "#04121f"
        border.width: 2
    }
}
