import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

CheckBox {
    id: ctl
    Layout.fillWidth: true
    font.pixelSize: Theme.fsBody
    implicitHeight: 28
    indicator: Rectangle {
        implicitWidth: 17; implicitHeight: 17
        x: 0
        y: (ctl.height - height) / 2
        radius: 5
        color: ctl.checked ? Theme.accent : Theme.panel
        border.color: ctl.checked ? Theme.accent : Theme.border
        border.width: 1
        Text {
            anchors.centerIn: parent
            visible: ctl.checked
            text: "✓"
            color: "#04121f"
            font.pixelSize: 11
            font.bold: true
        }
    }
    contentItem: Text {
        leftPadding: 24
        text: ctl.text
        color: Theme.textDim
        font: ctl.font
        verticalAlignment: Text.AlignVCenter
        wrapMode: Text.WordWrap
    }
}
