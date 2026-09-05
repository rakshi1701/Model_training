import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Terminal-style log pane that follows the tail unless the user scrolls up.
Rectangle {
    id: root
    property string text_: ""
    property bool follow: true
    Layout.fillWidth: true
    implicitHeight: 220
    radius: 10
    color: "#05080f"
    border.color: Theme.border
    border.width: 1
    clip: true

    ScrollView {
        id: scroller
        anchors.fill: parent
        anchors.margins: 8
        ScrollBar.horizontal.policy: ScrollBar.AsNeeded

        TextArea {
            id: area
            readOnly: true
            selectByMouse: true
            wrapMode: TextArea.NoWrap
            text: root.text_
            color: "#b9f5d8"
            font.family: Theme.mono
            font.pixelSize: Theme.fsSmall
            background: null
            // Follow the tail vertically only — moving the cursor would also
            // scroll the view sideways and hide the start of each line.
            onTextChanged: if (root.follow) Qt.callLater(function() {
                scroller.ScrollBar.vertical.position =
                    Math.max(0, 1 - scroller.ScrollBar.vertical.size)
                scroller.ScrollBar.horizontal.position = 0
            })
        }
    }
}
