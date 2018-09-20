import { Injectable } from "@angular/core";
import { Headers, Http, RequestOptions, Response } from "@angular/http";
import { Observable } from "rxjs";
import { map } from 'rxjs/operators';

@Injectable()
export class HomeService {

    constructor(private _http: Http) {

    }

    // getRouteData(): Observable<any> {
    //     return this._http
    //     .get("url")
    //     .pipe(map((res: Response) => res.json()))
    //     .catch(this.handleErrorObservable);
    // }

    /**
       * To handle the obervable error response
       * @param  {Response|any} error
       */
    private handleErrorObservable(error: Response | any) {
        return Observable.throw(error.message || error);
    }

}