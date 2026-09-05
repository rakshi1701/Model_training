import QtQuick
import QtQuick.Controls.Basic
import Studio

Switch {
    id: ctl
    font.pixelSize: Theme.fsBody
    indicator: Rectangle {
        implicitWidth: 40; implicitHeight: 20
        x: 0
        y: (ctl.height - height) / 2
        radius: 10
        color: ctl.checked ? Qt.rgba(0, 0.9, 0.64, 0.35) : Theme.panel
        border.color: ctl.checked ? Theme.success : Theme.border
        border.width: 1
        Rectangle {
            x: ctl.checked ? parent.width - width - 2 : 2
            y: 2
            width: 16; height: 16; radius: 8
            color: ctl.checked ? Theme.success : Theme.textMuted
            Behavior on x { NumberAnimation { duration: 120 } }
        }
    }
    contentItem: Text {
        leftPadding: 48
        text: ctl.text
        color: Theme.textDim
        font: ctl.font
        verticalAlignment: Text.AlignVCenter
    }
}
