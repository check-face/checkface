module BrowseFacesDialog
open Feliz
open Fable.MaterialUI.Icons
open Feliz.MaterialUI
open Fulma
open Browser.Types
open Browser
open Fable.Core.JsInterop
open Fetch
open Checkface
open Config
open Utils
open Feliz

type BrowseFacesDialogProps = {
    OnClose : unit -> unit
    IsOpen : bool
    RenderValue : CheckfaceSrc -> ReactElement
    OnValueSelected : CheckfaceSrc -> unit
    Values : CheckfaceSrc list option
}

let item renderValue onValueSelected value =
    Html.div [
        prop.onClick (fun _ -> onValueSelected value)
        prop.children [
            renderValue value
        ]
    ]

/// <summary>Parses data from an iframe message to a <see cref="CheckfaceSrc">CheckfaceSrc</see></summary>
/// <code>
/// // Inside the iframe, post a message like so
/// window.top.postMessage({ "value": "helloooo" }, "*") // CheckfaceValue
/// window.top.postMessage({ "seed": 1234 }, "*") // Seed
/// window.top.postMessage({ "guid": "54fec40f-12c4-4333-951b-6bc1d2d074b9" }, "*") // Guid
/// </code>
let parseMessageData (data:obj) = 
    // Data is untrusted and could be anything.
    // Check compiled javascript that casts and matches won't let any
    // unexpected data through

    let asGuid = 
        match data?guid:obj with
        | :? string as guidValue ->
            match guidValue |> (System.Guid.TryParse >> tryToOption) with
            | Some guid -> 
                Some (Guid guid)
            | None -> None
        | _ -> None

    let asSeed = 
        match data?seed:obj with
        //matching on uint will let floats through, so instead match on float and cast to uint. Casting to uint always returns a UInt32
        | :? float as seed -> Seed ((uint)seed) |> Some
        | _ -> None

    let asValue =
        match data?value:obj with
        | :? string as value -> CheckfaceValue value |> Some
        | _ -> None

    asGuid
    |> Option.orElse asSeed
    |> Option.orElse asValue


[<ReactComponent>]
let BrowseFacesDialog props =
    let frameWindow = React.useElementRef ()


    React.useEffect((fun () ->
        let onMessage (evt: Event) =
            match frameWindow.current with
            | None -> ()
            | Some element ->
                let frame = unbox<HTMLIFrameElement> element
                let message = evt :?> MessageEvent
                if message.source = (frame.contentWindow :> obj) then
                    match parseMessageData message.data with
                    | Some value -> props.OnValueSelected value
                    | None ->
                        console.error("Received malformed message from iframe", message)
                else
                    () // message is from a different iframe so ignore it

        window.addEventListener("message", onMessage)
        React.createDisposable(fun () -> window.removeEventListener("message", onMessage))
    ), [| props.OnValueSelected :> obj |])

    Mui.dialog [
        dialog.open' props.IsOpen
        dialog.fullScreen true
        dialog.onClose (fun _ _ -> props.OnClose ())
        dialog.children [
            Mui.dialogTitle [
                dialogTitle.children [
                    Html.div [
                        prop.style [ style.display.flex ]
                        prop.children [
                            Html.div "Select an Image"
                            Html.div [ 
                                prop.style [ style.flexGrow 1 ]
                            ]
                            Mui.iconButton [
                                prop.onClick (fun _ -> props.OnClose ())
                                iconButton.children [ closeIcon [ ] ]
                            ]
                        ]
                    ]
                ]
            ]
            Mui.dialogContent [
                prop.style [ style.padding 0; style.overflowY.visible; style.display.flex ]
                dialogContent.children [
                    if props.IsOpen && isSelfHost then
                        Html.div [
                            prop.style [ style.display.flex; style.flexWrap.wrap ]
                            prop.children [
                                for value in props.Values |> Option.defaultValue [] do
                                    item props.RenderValue props.OnValueSelected value
                            ]
                        ]
                    elif props.IsOpen then
                        Html.iframe [
                            prop.src browseFacesEmbedSrc
                            prop.style [ style.width (length.percent 100); style.height (length.percent 100) ]
                            prop.ref frameWindow
                        ]
                ]
            ]
        ]
    ]
