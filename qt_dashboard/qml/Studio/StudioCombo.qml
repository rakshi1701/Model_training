import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

ComboBox {
    id: ctl
    Layout.fillWidth: true
    implicitHeight: 34
    font.pixelSize: Theme.fsBody

    background: Rectangle {
        radius: 8
        color: Theme.panel
        border.color: ctl.activeFocus || ctl.hovered ? Theme.borderStrong : Theme.border
        border.width: 1
    }
    contentItem: Text {
        leftPadding: 10
        rightPadding: 28
        text: ctl.displayText
        color: Theme.text
        font: ctl.font
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    indicator: Text {
        x: ctl.width - width - 10
        y: (ctl.height - height) / 2
        text: "▾"
        color: Theme.textMuted
        font.pixelSize: 11
    }
    delegate: ItemDelegate {
        width: ctl.width
        height: 30
        highlighted: ctl.highlightedIndex === index
        contentItem: Text {
            text: modelData !== undefined ? modelData : ""
            color: highlighted ? Theme.accent : Theme.textDim
            font.pixelSize: Theme.fsBody
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            color: highlighted ? Theme.cardHover : "transparent"
        }
    }
    popup: Popup {
        y: ctl.height + 2
        width: ctl.width
        implicitHeight: Math.min(contentItem.implicitHeight + 8, 320)
        padding: 4
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: ctl.popup.visible ? ctl.delegateModel : null
            currentIndex: ctl.highlightedIndex
            ScrollBar.vertical: ScrollBar { }
        }
        background: Rectangle {
            radius: 8
            color: "#111c30"
            border.color: Theme.border
            border.width: 1
        }
    }
}
