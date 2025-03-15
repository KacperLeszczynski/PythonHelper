import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ChatModelStrategyEnum } from '../../shared/models/chat-model-strategy.enum';
import { ChatModelTypeEnum } from '../../shared/models/chat-model-type.enum';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = '/chat';
  private http = inject(HttpClient)

  getResponse(query: string, pythonVersion: string, numDocs: number, strategy: ChatModelStrategyEnum, chatModel: ChatModelTypeEnum): Observable<any> {
    let params = new HttpParams()
    .set('user_query', query)
    .set('python_version', pythonVersion)
    .set('num_docs', numDocs.toString())
    .set('strategy', strategy)
    .set('chat_model', chatModel);

    return this.http.get<any>(`${this.apiUrl}/`, { params });
  }
}
