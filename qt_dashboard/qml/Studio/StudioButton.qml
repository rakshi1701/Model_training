import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

Button {
    id: ctl
    property string kind: "secondary"   // primary | secondary | danger | ghost
    property color tint: kind === "primary" ? Theme.accent
                       : kind === "danger"  ? Theme.danger
                       : Theme.textDim
    implicitHeight: 34
    padding: 10
    leftPadding: 8
    rightPadding: 8
    font.pixelSize: Theme.fsBody
    font.bold: kind === "primary"
    opacity: enabled ? 1 : 0.4

    background: Rectangle {
        radius: 8
        color: kind === "primary"
               ? (ctl.down ? Qt.darker(Theme.accent, 1.3)
                           : ctl.hovered ? Qt.lighter(Theme.accent, 1.1) : Theme.accent)
               : kind === "ghost"
                 ? (ctl.hovered ? Theme.cardHover : "transparent")
                 : (ctl.down ? Qt.rgba(1,1,1,0.10)
                             : ctl.hovered ? Theme.cardHover : Theme.card)
        border.width: 1
        border.color: kind === "primary" ? "transparent"
                    : kind === "danger" ? Qt.rgba(0.94, 0.27, 0.27, 0.5)
                    : Theme.border
    }
    contentItem: Text {
        text: ctl.text
        font: ctl.font
        color: kind === "primary" ? "#04121f"
             : kind === "danger" ? Theme.danger : Theme.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
