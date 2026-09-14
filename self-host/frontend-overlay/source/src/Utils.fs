module Utils
open Fable.Core
open Fable.Core.JS
open Fable.Core.JsInterop
open Browser

let rec insert value index list =
    match index, list with
    | 0, xs -> value::xs
    | i, x::xs -> x::insert value (i - 1) xs
    | i, [] -> failwith "index out of range"

let rec remove i l =
    match i, l with
    | 0, x::xs -> xs
    | i, x::xs -> x::remove (i - 1) xs
    | i, [] -> failwith "index out of range"

let tryToOption (didSucceed, value) =
    if didSucceed then Some value else None

[<Emit("typeof gtag === 'function' && gtag('event', $0, { 'event_category': $1 })")>]
let gtagEvent action catagory : unit = jsNative

[<Emit("encodeURIComponent($0)")>]
let encodeUriComponent (str:string) : string = jsNative

type Clipboard =
    abstract writeText: string -> Promise<obj>
    abstract readText: unit -> Promise<string>

[<Emit("typeof $0 === 'undefined'")>]
let  isUndefined (x: 'a) : bool = jsNative

[<Emit("(!!$0)")>]
let isTruthy (x: 'a) : bool = jsNative

let clipboard : Clipboard option = if isUndefined navigator then None else navigator?clipboard

[<Emit("""typeof navigator !== 'undefined' && navigator.canShare && navigator.canShare($0)""")>]
let navigatorCanShareThing (data:Types.ShareData) : bool = jsNative
let navigatorCanShare = navigatorCanShareThing !!{| url = "https://example.com"; text = "examplewwe text"; title = "nonononono" |}
printfn "Can share? %A" navigatorCanShare
let navigatorShare url title text = navigator.share !!{| url = url; title = title; text = text |}

let eventTargetElementValue (eventTarget:Types.EventTarget) : Types.Element = eventTarget?value

type FileShare = {
    FileUrl : string
    ContentType : string
    FileName : string
    Title : string
    Url : string
    Text : string
}

let shareFileFromUrl : FileShare -> Promise<bool> = import "shareFile" "./shareFile.js"
