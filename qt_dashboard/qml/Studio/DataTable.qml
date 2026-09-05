import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Lightweight table: `columns` is [{title, key, width}], `rows` an array of maps.
Rectangle {
    id: root
    property var columns: []
    property var rows: []
    property int rowHeight: 30
    property int maxHeight: 320

    Layout.fillWidth: true
    radius: 10
    color: Theme.card
    border.color: Theme.border
    border.width: 1
    clip: true
    implicitHeight: Math.min(maxHeight, rowHeight * ((rows ? rows.length : 0) + 1) + 12)

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 5
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: root.rowHeight
            spacing: 6
            Repeater {
                model: root.columns
                delegate: Text {
                    required property var modelData
                    text: modelData.title
                    color: Theme.accent
                    font.pixelSize: Theme.fsSmall
                    font.bold: true
                    leftPadding: 6
                    elide: Text.ElideRight
                    verticalAlignment: Text.AlignVCenter
                    Layout.fillWidth: modelData.width === undefined
                    Layout.preferredWidth: modelData.width !== undefined ? modelData.width : -1
                }
            }
        }
        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: Theme.border }

        ListView {
            id: list
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: root.rows
            ScrollBar.vertical: ScrollBar { }
            delegate: Rectangle {
                id: rowItem
                required property int index
                required property var modelData
                width: list.width
                height: root.rowHeight
                color: index % 2 ? Qt.rgba(1, 1, 1, 0.02) : "transparent"
                RowLayout {
                    anchors.fill: parent
                    spacing: 6
                    Repeater {
                        model: root.columns
                        delegate: Text {
                            required property var modelData
                            text: {
                                var v = rowItem.modelData[modelData.key]
                                return (v === undefined || v === null) ? "—" : String(v)
                            }
                            color: Theme.textDim
                            font.pixelSize: Theme.fsSmall
                            leftPadding: 6
                            elide: Text.ElideRight
                            verticalAlignment: Text.AlignVCenter
                            Layout.fillWidth: modelData.width === undefined
                            Layout.preferredWidth: modelData.width !== undefined ? modelData.width : -1
                        }
                    }
                }
            }
        }
    }
}
